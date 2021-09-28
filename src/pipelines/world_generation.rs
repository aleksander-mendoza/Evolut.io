use crate::render::stage_buffer::{StageBuffer, StageSubBuffer, IndirectDispatchSubBuffer, IndirectSubBuffer};
use crate::pipelines::particle::Particle;
use crate::render::command_pool::{CommandPool, CommandBuffer};


use ash::vk;


use failure::Error;


use crate::render::submitter::{Submitter, fill_submit, fill_zeros_submit};

use crate::render::buffer_type::{Cpu, Storage, GpuIndirect, Uniform};

use crate::blocks::world_size::{CHUNK_VOLUME_IN_CELLS, CHUNK_WIDTH, CHUNK_DEPTH, BROAD_PHASE_CHUNK_VOLUME_IN_CELLS, BROAD_PHASE_CELL_CAPACITY};
use crate::render::subbuffer::SubBuffer;
use crate::pipelines::constraint::Constraint;
use crate::render::buffer::{Buffer, make_shader_buffer_barrier};
use crate::pipelines::global_mutables::{GlobalMutables};
use crate::blocks::{WorldSize, Block, Face, WorldBlocks, WorldFaces};
use crate::render::sampler::Sampler;
use crate::pipelines::bone::Bone;
use crate::blocks::block_properties::{BLOCKS, BlockProp, BEDROCK, DIRT, GRASS, GLASS, PLANK, AIR, STONE, WATER};
use crate::pipelines::sensor::Sensor;
use crate::pipelines::neural_net_layer::NeuralNetLayer;
use crate::pipelines::neural_net_layer::Aggregate::Overwrite;
use crate::pipelines::muscle::Muscle;
use crate::neat::neat::Neat;
use crate::neat::num::Num;
use crate::neat::cppn::CPPN;
use crate::render::device::{QUEUE_IDX_GRAPHICS, QUEUE_IDX_TRANSFER};
use crate::render::host_buffer::HostBuffer;
use crate::pipelines::player_event::PlayerEvent;
use crate::render::compute::{ComputeDescriptorsBuilder, ComputeDescriptors};
use crate::render::specialization_constants::SpecializationConstants;
use crate::neat::htm_entity::{HtmEntity, ENTITY_MAX_LIDAR_COUNT};
use crate::neat::ann_entity::AnnEntity;
use crate::pipelines::foundations::{FoundationInitializer, Foundations};
use crate::pipelines::perlin_noise_map::{PerlinNoiseMap, RandomMap};
use crate::render::shader_module::{ShaderModule, Compute};
use crate::render::fence::Fence;

pub const SEA_LEVEL:u32 = 128;
pub const FREEZING_TEMPERATURE:f32 = 0.;
pub const SWAMP_HUMIDITY:f32 = 40.;
pub const DESERT_HUMIDITY:f32 = 25.;//humidity of sahara desert
pub const LARGE_SCALE:f32 = 64.;
pub const CHUNK_SCALE:f32 = 16.;
pub const TEMPERATURE_SCALE:f32 = 64.;
pub const HUMIDITY_SCALE:f32 = 64.;
pub const RESOURCE_TYPE_SCALE:f32 = 16.;
pub const HAS_RESOURCE_SCALE:f32 = 4.;
pub struct WorldGeneratorInitializer {
    large_scale:Submitter<StageSubBuffer<f32,Cpu,Storage>>,
    chunk_scale:Submitter<StageSubBuffer<f32,Cpu,Storage>>,
    temperature_scale:Submitter<StageSubBuffer<f32,Cpu,Storage>>,
    humidity_scale:Submitter<StageSubBuffer<f32,Cpu,Storage>>,
    resource_type_scale:Submitter<StageSubBuffer<f32,Cpu,Storage>>,
    has_resource_scale:Submitter<StageSubBuffer<f32,Cpu,Storage>>,
    surface_artifact_scale:Submitter<StageSubBuffer<f32,Cpu,Storage>>,
    random_vals_buffer:SubBuffer<f32,Storage>,
    generate_world_stone_dirt_grass:ShaderModule<Compute>,
    generate_world_face_count_per_chunk:ShaderModule<Compute>,
    generate_world_faces_prepare_face_offsets:ShaderModule<Compute>,
    generate_world_faces:ShaderModule<Compute>,
    generate_world_copy:ShaderModule<Compute>,
    generate_world_update_meta:ShaderModule<Compute>,
    generate_world_agents:ShaderModule<Compute>,
    generate_world_rand_uint:ShaderModule<Compute>,
}


impl WorldGeneratorInitializer {
    pub fn new(cmd_pool: &CommandPool,foundations: &FoundationInitializer) -> Result<Self, failure::Error> {
        let world_size = foundations.world_size().clone();
        let large_scale = RandomMap::new_around(world_size, LARGE_SCALE, 128., 80.);
        let chunk_scale = RandomMap::new_around(world_size, CHUNK_SCALE, 0., 16.);
        let temperature_scale = RandomMap::new_between(world_size, TEMPERATURE_SCALE, -30., 70.); // degree celsius;
        let humidity_scale = RandomMap::new_between(world_size, HUMIDITY_SCALE, 0., 80.); // percentage;
        let resource_type_scale = RandomMap::new_between(world_size, RESOURCE_TYPE_SCALE, 0., 1.);
        let has_resource_scale = RandomMap::new_between(world_size, HAS_RESOURCE_SCALE, 0., 1.);
        let surface_artifact_scale:Vec<f32> = (0..world_size.world_area()).map(|_|f32::random()).collect();
        let generate_world_stone_dirt_grass = ShaderModule::new(include_glsl!("assets/shaders/generate_world_stone_dirt_grass.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let generate_world_face_count_per_chunk = ShaderModule::new(include_glsl!("assets/shaders/generate_world_face_count_per_chunk.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let generate_world_faces_prepare_face_offsets = ShaderModule::new(include_glsl!("assets/shaders/generate_world_faces_prepare_face_offsets.comp", target: vulkan1_1, kind: comp) as &[u32], cmd_pool.device())?;
        let generate_world_faces = ShaderModule::new(include_glsl!("assets/shaders/generate_world_faces.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let generate_world_copy = ShaderModule::new(include_glsl!("assets/shaders/generate_world_copy.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let generate_world_update_meta = ShaderModule::new(include_glsl!("assets/shaders/generate_world_update_meta.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let generate_world_agents = ShaderModule::new(include_glsl!("assets/shaders/generate_world_agents.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let generate_world_rand_uint = ShaderModule::new(include_glsl!("assets/shaders/generate_world_rand_uint.comp", kind: comp) as &[u32], cmd_pool.device())?;
        // heights.setup_world_blocks(&mut data.world_blocks);
        // data.setup_world_faces();
        // for _ in 0..8 {
        //     let x = rand::random::<usize>() % cap.world_size.world_width();
        //     let z = rand::random::<usize>() % cap.world_size.world_depth();
        //     data.bone_data.push(Bone::new(glm::vec3(x as f32, heights.height(x, z) as f32 + 5., z as f32), 0.5, f32::random_vec3(), 1., 1.0));
        // }
        // let mutables = data.compute_constants(&cap);

        let surface_artifact_scale_in_bytes = std::mem::size_of_val(surface_artifact_scale.as_slice()) as ash::vk::DeviceSize;

        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            large_scale.byte_len() +
                                                                                chunk_scale.byte_len() +
                                                                                temperature_scale.byte_len() +
                                                                                humidity_scale.byte_len() +
                                                                                resource_type_scale.byte_len() +
                                                                                has_resource_scale.byte_len() +
                                                                                surface_artifact_scale_in_bytes,
        )?;

        let offset = 0 as vk::DeviceSize;
        let large_scale_buffer = super_buffer.sub(offset..offset + large_scale.byte_len()).reinterpret_into::<f32>();
        let offset = offset + large_scale.byte_len();
        let large_scale = StageBuffer::wrap(cmd_pool, large_scale.as_slice(), large_scale_buffer)?;
        let chunk_scale_buffer = super_buffer.sub(offset..offset + chunk_scale.byte_len()).reinterpret_into::<f32>();
        let offset = offset + chunk_scale.byte_len();
        let chunk_scale = StageBuffer::wrap(cmd_pool, chunk_scale.as_slice(), chunk_scale_buffer)?;
        let temperature_scale_buffer = super_buffer.sub(offset..offset + temperature_scale.byte_len()).reinterpret_into::<f32>();
        let offset = offset + temperature_scale.byte_len();
        let temperature_scale = StageBuffer::wrap(cmd_pool, temperature_scale.as_slice(), temperature_scale_buffer)?;
        let humidity_scale_buffer = super_buffer.sub(offset..offset + humidity_scale.byte_len()).reinterpret_into::<f32>();
        let offset = offset + humidity_scale.byte_len();
        let humidity_scale = StageBuffer::wrap(cmd_pool, humidity_scale.as_slice(), humidity_scale_buffer)?;
        let resource_type_scale_buffer = super_buffer.sub(offset..offset + resource_type_scale.byte_len()).reinterpret_into::<f32>();
        let offset = offset + resource_type_scale.byte_len();
        let resource_type_scale = StageBuffer::wrap(cmd_pool, resource_type_scale.as_slice(), resource_type_scale_buffer)?;
        let has_resource_scale_buffer = super_buffer.sub(offset..offset + has_resource_scale.byte_len()).reinterpret_into::<f32>();
        let offset = offset + has_resource_scale.byte_len();
        let has_resource_scale = StageBuffer::wrap(cmd_pool, has_resource_scale.as_slice(), has_resource_scale_buffer)?;
        let surface_artifact_scale_buffer = super_buffer.sub(offset..offset + surface_artifact_scale_in_bytes).reinterpret_into::<f32>();
        let offset = offset + surface_artifact_scale_in_bytes;
        let surface_artifact_scale = StageBuffer::wrap(cmd_pool, surface_artifact_scale.as_slice(), surface_artifact_scale_buffer)?;

        let random_vals_buffer = super_buffer.reinterpret_into::<f32>();
        Ok(Self {
            surface_artifact_scale,
            generate_world_stone_dirt_grass,
            large_scale,
            chunk_scale,
            temperature_scale,
            humidity_scale,
            resource_type_scale,
            has_resource_scale,
            generate_world_face_count_per_chunk,
            generate_world_faces_prepare_face_offsets,
            generate_world_faces,
            generate_world_copy,
            random_vals_buffer,
            generate_world_update_meta,
            generate_world_agents,
            generate_world_rand_uint,
        })
    }
    pub fn build(self, cmd_pool: &CommandPool, foundations:&Foundations) -> Result<(), Error> {
        let Self {
            generate_world_update_meta,
            generate_world_stone_dirt_grass,
            surface_artifact_scale,
            large_scale,
            chunk_scale,
            temperature_scale,
            humidity_scale,
            resource_type_scale,
            has_resource_scale,
            random_vals_buffer ,
            generate_world_copy,
            generate_world_face_count_per_chunk,
            generate_world_faces_prepare_face_offsets,
            generate_world_faces,
            generate_world_agents,
            generate_world_rand_uint
        } = self;
        let mut descriptors = ComputeDescriptorsBuilder::new();
        descriptors.storage_buffer(foundations.global_mutables());//0
        descriptors.storage_buffer(foundations.world_block_meta());//1
        descriptors.storage_buffer(foundations.indirect().super_buffer());//2
        descriptors.storage_buffer(foundations.bones());//3
        descriptors.storage_buffer(foundations.world());//4
        descriptors.storage_buffer(foundations.world_copy());//5
        descriptors.storage_buffer(foundations.faces());//6
        descriptors.storage_buffer(&random_vals_buffer);//7
        descriptors.storage_buffer(foundations.rand_uint());//8
        descriptors.storage_buffer(foundations.ann_entities_buffer());//9
        let descriptors = descriptors.build(cmd_pool.device())?;
        let large_scale = large_scale.take()?.take_gpu();
        let chunk_scale = chunk_scale.take()?.take_gpu();
        let temperature_scale = temperature_scale.take()?.take_gpu();
        let humidity_scale = humidity_scale.take()?.take_gpu();
        let resource_type_scale = resource_type_scale.take()?.take_gpu();
        let has_resource_scale = has_resource_scale.take()?.take_gpu();
        let surface_artifact_scale = surface_artifact_scale.take()?.take_gpu();
        let sc = foundations.specialization_constants().build();
        let generate_world_stone_dirt_grass = descriptors.build("main", generate_world_stone_dirt_grass,&sc)?;
        let generate_world_face_count_per_chunk =  descriptors.build("main", generate_world_face_count_per_chunk,&sc)?;
        let generate_world_faces_prepare_face_offsets = descriptors.build("main", generate_world_faces_prepare_face_offsets,&sc)?;
        let generate_world_faces = descriptors.build("main", generate_world_faces,&sc)?;
        let generate_world_copy = descriptors.build("main", generate_world_copy,&sc)?;
        let generate_world_update_meta = descriptors.build("main", generate_world_update_meta,&sc)?;
        let generate_world_agents = descriptors.build("main", generate_world_agents,&sc)?;
        let generate_world_rand_uint = descriptors.build("main", generate_world_rand_uint,&sc)?;
        let mut generate_command_buffer = cmd_pool.create_command_buffer()?;
        let subgroup_size = cmd_pool.device().get_max_subgroup_size();
        let world_area = foundations.world_size().world_area() as u32;
        let world_volume = foundations.world_size().world_volume() ;
        let agents_groups = foundations.default_global_mutables().ann_entities/subgroup_size;
        assert_eq!(world_area%subgroup_size,0,"World area {} is not divisible by subgroup size {}",world_area,cmd_pool.device().get_max_subgroup_size());
        assert!(foundations.default_global_mutables().ann_entities<foundations.cap().max_rand_uint as u32/ /*xz dimensions*/2);

        let area_groups = world_area/subgroup_size;
        let rand_uint_groups = (foundations.cap().max_rand_uint as u32+subgroup_size-1)/subgroup_size;
        let volume_groups = (world_volume/subgroup_size as usize) as u32;
        generate_command_buffer.reset()?
            .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?
            .bind_compute_descriptors(&generate_world_stone_dirt_grass)
            .bind_compute_pipeline(&generate_world_stone_dirt_grass)
            .dispatch_1d(area_groups)
            .bind_compute_pipeline(&generate_world_rand_uint)
            .dispatch_1d(rand_uint_groups)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.world()),
                make_shader_buffer_barrier(foundations.world_block_meta()),
                make_shader_buffer_barrier(foundations.rand_uint()),
            ])
            // .bind_compute_pipeline(&generate_world_update_meta)
            // .dispatch_1d(volume_groups)
            // .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
            //     make_shader_buffer_barrier(foundations.world())
            // ])
            .bind_compute_pipeline(&generate_world_face_count_per_chunk)
            .dispatch_1d(area_groups)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.faces())
            ])
            .bind_compute_pipeline(&generate_world_faces_prepare_face_offsets)
            .dispatch_1d(1)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.faces())
            ])
            .bind_compute_pipeline(&generate_world_agents)
            .dispatch_1d(agents_groups)
            .bind_compute_pipeline(&generate_world_faces)
            .dispatch_1d(area_groups)
            .bind_compute_pipeline(&generate_world_copy)
            .dispatch_1d(volume_groups)
            .end()?;
        let generate_fence = Fence::new(cmd_pool.device(), false)?;
        generate_command_buffer.submit(&[], &[], Some(&generate_fence))?;
        generate_fence.wait(None)?;
        Ok(())
    }
}
