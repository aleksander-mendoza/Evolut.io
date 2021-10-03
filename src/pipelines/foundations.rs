use crate::render::stage_buffer::{StageBuffer, StageSubBuffer, IndirectDispatchSubBuffer, IndirectSubBuffer};
use crate::pipelines::particle::Particle;
use crate::render::command_pool::{CommandPool};


use ash::vk;


use failure::Error;


use crate::render::submitter::{Submitter, fill_submit, fill_zeros_submit};

use crate::render::buffer_type::{Cpu, Storage, GpuIndirect, Uniform};

use crate::blocks::world_size::{CHUNK_VOLUME_IN_CELLS, CHUNK_WIDTH, CHUNK_DEPTH, BROAD_PHASE_CHUNK_VOLUME_IN_CELLS, BROAD_PHASE_CELL_CAPACITY};
use crate::render::subbuffer::SubBuffer;
use crate::pipelines::constraint::Constraint;
use crate::render::buffer::Buffer;
use crate::pipelines::global_mutables::{GlobalMutables};
use crate::blocks::{WorldSize, Block, Face, BlockMeta};
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
use std::time::{UNIX_EPOCH, SystemTime};
use crate::neat::ann_entity::AnnEntity;

pub struct Indirect {
    per_particle: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    per_bone: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    agent_sensory_input_update: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    update_ambience: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    per_blocks_to_be_inserted_or_removed: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    per_htm_entity: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    per_ann_entity: SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>,
    draw_bones: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_blocks: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    draw_particles: SubBuffer<vk::DrawIndirectCommand, GpuIndirect>,
    super_indirect_buffer: SubBuffer<u8, GpuIndirect>,
}

impl Indirect {
    fn new(super_indirect_buffer: SubBuffer<u8, GpuIndirect>, indirect_dispatch: &Submitter<IndirectDispatchSubBuffer>, indirect_draw: &Submitter<IndirectSubBuffer>) -> Self {
        let per_particle = indirect_dispatch.gpu().element(0);
        let per_bone = indirect_dispatch.gpu().element(1);
        let agent_sensory_input_update = indirect_dispatch.gpu().element(2);
        let update_ambience = indirect_dispatch.gpu().element(3);
        let per_blocks_to_be_inserted_or_removed = indirect_dispatch.gpu().element(4);
        let per_htm_entity = indirect_dispatch.gpu().element(5);
        let per_ann_entity = indirect_dispatch.gpu().element(6);

        let draw_bones = indirect_draw.gpu().element(0);
        let draw_blocks = indirect_draw.gpu().element(1);
        let draw_particles = indirect_draw.gpu().element(2);
        Self {
            per_htm_entity,
            per_ann_entity,
            super_indirect_buffer,
            per_particle,
            per_bone,
            agent_sensory_input_update,
            draw_bones,
            draw_blocks,
            draw_particles,
            update_ambience,
            per_blocks_to_be_inserted_or_removed,
        }
    }
    pub fn update_htm_entities(&self)->&SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>{
        &self.per_htm_entity
    }
    pub fn update_ann_entities(&self)->&SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>{
        &self.per_ann_entity
    }
    pub fn update_ambience(&self)->&SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>{
        &self.update_ambience
    }
    pub fn update_ambience_faces(&self)->&SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>{
        &self.per_blocks_to_be_inserted_or_removed
    }
    pub fn update_ambience_flush_world_copy(&self)->&SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>{
        &self.per_blocks_to_be_inserted_or_removed
    }
    pub fn super_buffer(&self) -> &SubBuffer<u8, GpuIndirect> {
        &self.super_indirect_buffer
    }
    pub fn draw_bones(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_bones
    }
    pub fn draw_blocks(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_blocks
    }
    pub fn draw_particles(&self) -> &SubBuffer<vk::DrawIndirectCommand, GpuIndirect> {
        &self.draw_particles
    }
    pub fn update_particles(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_particle
    }
    pub fn broad_phase_collision_detection(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_bone
    }
    pub fn narrow_phase_collision_detection(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_bone
    }
    pub fn broad_phase_collision_detection_cleanup(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_bone
    }
    pub fn agent_sensory_input_update(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.agent_sensory_input_update
    }
    pub fn update_bones(&self) -> &SubBuffer<vk::DispatchIndirectCommand, GpuIndirect> {
        &self.per_bone
    }
}

pub struct FoundationInitializer {
    cap:FoundationsCapacity,
    specialization_constants: SpecializationConstants,
    world: SubBuffer<Block, Storage>,
    faces: SubBuffer<Face, Storage>,
    rand_uint: SubBuffer<u32, Storage>,
    face_count_per_chunk_buffer: SubBuffer<Face, Storage>,
    htm_entities_buffer: SubBuffer<HtmEntity, Storage>,
    ann_entities_buffer: SubBuffer<AnnEntity, Storage>,
    opaque_and_transparent_face_buffer: SubBuffer<Face, Storage>,
    tmp_faces_copy: Submitter<SubBuffer<u32, Storage>>,
    blocks_to_be_inserted_or_removed: SubBuffer<u32, Storage>,
    faces_to_be_inserted: SubBuffer<Face, Storage>,
    faces_to_be_removed: SubBuffer<u32, Storage>,
    player_event_uniform: HostBuffer<PlayerEvent, Uniform>,
    collision_grid: SubBuffer<u32, Storage>,
    bones: SubBuffer<Bone, Storage>,
    default_global_mutables:GlobalMutables,
    particles: SubBuffer<Particle, Storage>,
    global_mutables: Submitter<StageSubBuffer<GlobalMutables, Cpu, Storage>>,
    indirect_dispatch: Submitter<IndirectDispatchSubBuffer>,
    indirect_draw: Submitter<IndirectSubBuffer>,
    indirect: Indirect,
    sampler: Sampler,
}

pub struct FoundationsCapacity {
    pub world_size: WorldSize,
    pub faces_to_be_inserted_chunk_capacity: u32,
    pub faces_to_be_removed_chunk_capacity: u32,
    pub max_bones: u64,
    pub max_faces: u64,
    pub max_faces_copy: u64,
    pub max_rand_uint: u64,
    pub max_tmp_faces_copy: u64,
    pub max_blocks_to_be_inserted_or_removed: u64,
    pub max_faces_to_be_inserted: u64,
    pub max_faces_to_be_removed: u64,
    pub max_sensors: u64,
    pub max_htm_entities: u64,
    pub max_particles: u64,
    pub max_ann_entities: u64,
}

impl FoundationsCapacity {
    pub fn new(x: usize, z: usize) -> Self {
        let world_size = WorldSize::new(x, z);
        let faces_to_be_inserted_chunk_capacity = 128;
        let faces_to_be_removed_chunk_capacity = 128;
        Self {
            faces_to_be_inserted_chunk_capacity,
            faces_to_be_removed_chunk_capacity,
            max_bones: 1024u64,
            max_faces: 16 * 1024u64 * world_size.total_chunks() as u64,
            max_rand_uint: 64 * 1024u64, // used as backing memory for vectors, matrices and
            // tensors that make up various neural networks. Especially, the outputs of recursive neural networks
            // often need to be persistent, because those outputs are later fed as inputs to the same neural net.
            max_tmp_faces_copy: world_size.world_volume() as u64 / 4,
            max_blocks_to_be_inserted_or_removed: world_size.world_volume() as u64 / 16,
            max_faces_to_be_inserted: faces_to_be_inserted_chunk_capacity as u64 * 2 * world_size.total_chunks() as u64,
            max_faces_to_be_removed: faces_to_be_removed_chunk_capacity as u64 * 2 * world_size.total_chunks() as u64,
            max_sensors: 0u64,
            max_htm_entities: 128u64,
            max_ann_entities: 2048u64,
            max_faces_copy: 1024u64 * world_size.total_chunks() as u64,
            max_particles: 1024u64,
            world_size,
        }
    }
    fn grid_size(&self) -> u64 {
        (self.world_size.total_chunks() * BROAD_PHASE_CHUNK_VOLUME_IN_CELLS) as u64
    }
}

struct CollisionCell{
    len:u32,
    contents:[u32;BROAD_PHASE_CELL_CAPACITY]
}
fn append_owned<X>(v: &mut Vec<X>, mut v2: Vec<X>) {
    v.append(&mut v2);
}

impl FoundationInitializer {
    pub fn faces(&self) -> &SubBuffer<Face, Storage> {
        &self.faces
    }

    pub fn face_count_per_chunk_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.opaque_and_transparent_face_buffer
    }
    pub fn tmp_faces_copy(&self) -> &SubBuffer<u32, Storage> {
        &self.tmp_faces_copy
    }
    pub fn global_mutables(&self) -> &StageSubBuffer<GlobalMutables, Cpu, Storage> {
        &self.global_mutables
    }
    pub fn specialization_constants(&self) -> &SpecializationConstants{
        &self.specialization_constants
    }
    pub fn particles(&self) -> &SubBuffer<Particle, Storage> {
        &self.particles
    }
    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
    pub fn bones(&self) -> &SubBuffer<Bone, Storage> {
        &self.bones
    }
    pub fn cap(&self) -> &FoundationsCapacity {
        &self.cap
    }
    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }
    pub fn world_size(&self) -> &WorldSize {
        &self.cap.world_size
    }
    pub fn world(&self) -> &SubBuffer<Block, Storage> {
        &self.world
    }

    pub fn new(cmd_pool: &CommandPool) -> Result<Self, failure::Error> {
        let cap = FoundationsCapacity::new(16,16);
        let entity_count = 8*cap.world_size.total_chunks() as u32;
        let mutables =  GlobalMutables {
            blocks_to_be_inserted_or_removed: 0,
            bones: entity_count,
            particles: 0,
            held_bone_idx: 0,
            htm_entities: 0,
            tick: 0,
            lidars: 0,
            ann_entities: entity_count,
        };
        assert!(cap.max_ann_entities>=entity_count as u64, "{} >= {}", cap.max_ann_entities,entity_count);
        let bones_in_bytes = std::mem::size_of::<Bone>() as u64 * cap.max_bones;
        let faces_in_bytes = std::mem::size_of::<Face>() as u64 * cap.max_faces;
        let tmp_faces_copy_in_bytes = std::mem::size_of::<u32>() as u64 * 3 * cap.max_faces_copy;
        let grid_in_bytes = std::mem::size_of::<CollisionCell>() as u64 * cap.grid_size();
        let world_in_bytes = (std::mem::size_of::<Block>() * cap.world_size.world_volume()) as u64;
        let blocks_to_be_inserted_or_removed_in_bytes = std::mem::size_of::<u32>() as u64 * cap.max_blocks_to_be_inserted_or_removed;
        let global_mutables_in_bytes = std::mem::size_of_val(&mutables) as u64;
        let rand_uint_in_bytes = std::mem::size_of::<f32>() as u64 * cap.max_rand_uint;
        let htm_entities_in_bytes = std::mem::size_of::<HtmEntity>() as u64 * cap.max_htm_entities;
        let ann_entities_in_bytes = std::mem::size_of::<AnnEntity>() as u64 * cap.max_ann_entities;
        let faces_to_be_inserted_in_bytes = std::mem::size_of::<Face>() as u64 * cap.max_faces_to_be_inserted;
        let faces_to_be_removed_in_bytes = std::mem::size_of::<u32>() as u64 * cap.max_faces_to_be_removed;
        let particles_in_bytes = std::mem::size_of::<Particle>() as u64 * cap.max_particles;

        let super_buffer: SubBuffer<u8, Storage> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                            bones_in_bytes +
                                                                                faces_in_bytes +
                                                                                tmp_faces_copy_in_bytes +
                                                                                grid_in_bytes +
                                                                                world_in_bytes +
                                                                                blocks_to_be_inserted_or_removed_in_bytes +
                                                                                global_mutables_in_bytes +
                                                                                particles_in_bytes +
                                                                                faces_to_be_inserted_in_bytes +
                                                                                faces_to_be_removed_in_bytes +
                                                                                htm_entities_in_bytes +
                                                                                ann_entities_in_bytes +
                                                                                rand_uint_in_bytes
        )?;
        let offset = 0;
        let bones_buffer = super_buffer.sub(offset..offset + bones_in_bytes).reinterpret_into::<Bone>();
        let offset = offset + bones_in_bytes;
        assert_eq!(offset % 16, 0);
        let face_buffer = super_buffer.sub(offset..offset + faces_in_bytes).reinterpret_into::<Face>();
        let offset = offset + faces_in_bytes ;
        assert_eq!(offset % 16, 0);
        let tmp_faces_copy_buffer = super_buffer.sub(offset..offset + tmp_faces_copy_in_bytes).reinterpret_into::<u32>();
        let offset = offset + tmp_faces_copy_in_bytes;
        assert_eq!(offset % 16, 0);
        let grid_buffer = super_buffer.sub(offset..offset + grid_in_bytes).reinterpret_into::<u32>();
        let offset = offset + grid_in_bytes;
        assert_eq!(offset % 16, 0);
        let world_buffer = super_buffer.sub(offset..offset + world_in_bytes).reinterpret_into::<Block>();
        let offset = offset + world_in_bytes;
        assert_eq!(offset % 16, 0);
        let blocks_to_be_inserted_or_removed_buffer = super_buffer.sub(offset..offset + blocks_to_be_inserted_or_removed_in_bytes).reinterpret_into::<u32>();
        let offset = offset + blocks_to_be_inserted_or_removed_in_bytes;
        assert_eq!(offset % 16, 0);
        let global_mutables_buffer = super_buffer.sub(offset..offset + global_mutables_in_bytes).reinterpret_into::<GlobalMutables>();
        let offset = offset + global_mutables_in_bytes;
        assert_eq!(offset % 16, 0);
        let particles_buffer = super_buffer.sub(offset..offset + particles_in_bytes).reinterpret_into::<Particle>();
        let offset = offset + particles_in_bytes;
        assert_eq!(offset % 16, 0);
        let faces_to_be_inserted_buffer = super_buffer.sub(offset..offset + faces_to_be_inserted_in_bytes).reinterpret_into::<Face>();
        let offset = offset + faces_to_be_inserted_in_bytes;
        assert_eq!(offset % 16, 0);
        let faces_to_be_removed_buffer = super_buffer.sub(offset..offset + faces_to_be_removed_in_bytes).reinterpret_into::<u32>();
        let offset = offset + faces_to_be_removed_in_bytes;
        assert_eq!(offset % 16, 0);
        let htm_entities_buffer = super_buffer.sub(offset..offset + htm_entities_in_bytes).reinterpret_into::<HtmEntity>();
        let offset = offset + htm_entities_in_bytes;
        assert_eq!(offset % 16, 0);
        let ann_entities_buffer = super_buffer.sub(offset..offset + ann_entities_in_bytes).reinterpret_into::<AnnEntity>();
        let offset = offset + ann_entities_in_bytes;
        assert_eq!(offset % 16, 0);
        let rand_uint_buffer = super_buffer.sub(offset..offset + rand_uint_in_bytes).reinterpret_into::<u32>();
        let offset = offset + rand_uint_in_bytes;
        assert_eq!(offset % 16, 0);

        let mut tmp_faces_copy = Submitter::new(tmp_faces_copy_buffer, cmd_pool)?;
        fill_zeros_submit(&mut tmp_faces_copy)?;

        let global_mutables = StageBuffer::wrap(cmd_pool, &[mutables], global_mutables_buffer)?;

        let face_count_per_chunk_buffer = face_buffer.sub(..std::mem::size_of::<Face>() as u64 * cap.world_size.total_chunks() as u64 * 2);
        let opaque_and_transparent_face_buffer = face_buffer.sub(std::mem::size_of::<Face>() as u64 * cap.world_size.total_chunks() as u64 * 2..);

        let particles = particles_buffer;

        let sampler = Sampler::new(cmd_pool.device(), vk::Filter::NEAREST, true)?;

        fn dispatch_indirect(x: usize, group_size:u32) -> vk::DispatchIndirectCommand {
            vk::DispatchIndirectCommand {
                x: (x as f32 / group_size as f32).ceil() as u32,
                y: 1,
                z: 1,
            }
        }
        fn draw_indirect(vertex_count: u32, instance_count: u32) -> vk::DrawIndirectCommand {
            vk::DrawIndirectCommand {
                vertex_count,
                instance_count,
                first_vertex: 0,
                first_instance: 0,
            }
        }
        let indirect_dispatch_data = vec![
            dispatch_indirect(0, cmd_pool.device().get_max_subgroup_size()),// update_particles.comp
            dispatch_indirect(mutables.bones as usize, cmd_pool.device().get_max_subgroup_size()),// broad_phase_collision_detection.comp broad_phase_collision_detection_cleanup.comp narrow_phase_collision_detection.comp update_bones.comp
            dispatch_indirect(0, cmd_pool.device().get_max_subgroup_size()), // agent_sensory_input_update.comp
            dispatch_indirect(0, cmd_pool.device().get_max_subgroup_size()),// update_ambience.comp
            dispatch_indirect(0, cmd_pool.device().get_max_subgroup_size()),// update_ambience_faces.comp, update_ambience_flush_world_copy.comp
            dispatch_indirect(0, cmd_pool.device().get_max_subgroup_size()),// update_htm_entities.comp
            vk::DispatchIndirectCommand{
                x: mutables.ann_entities,
                y: 1,
                z: 1
            },// update_ann_entities.comp
        ];
        let indirect_draw_data = vec![
            draw_indirect(36, mutables.bones),// bones.vert
            draw_indirect(6, 0),// block.vert
            draw_indirect(0, 0),// particles.vert
        ];
        let indirect_dispatch_in_bytes = std::mem::size_of_val(indirect_dispatch_data.as_slice()) as u64;
        let indirect_draw_in_bytes = std::mem::size_of_val(indirect_draw_data.as_slice()) as u64;
        let super_indirect_buffer: SubBuffer<u8, GpuIndirect> = SubBuffer::with_capacity(cmd_pool.device(),
                                                                                         indirect_dispatch_in_bytes +
                                                                                             indirect_draw_in_bytes)?;
        let offset = 0;
        let indirect_dispatch_buffer = super_indirect_buffer.sub(offset..offset + indirect_dispatch_in_bytes).reinterpret_into::<vk::DispatchIndirectCommand>();
        let offset = offset + indirect_dispatch_in_bytes;
        let indirect_draw_buffer = super_indirect_buffer.sub(offset..offset + indirect_draw_in_bytes).reinterpret_into::<vk::DrawIndirectCommand>();
        let offset = offset + indirect_draw_in_bytes;

        let indirect_dispatch = StageBuffer::wrap(cmd_pool, &indirect_dispatch_data, indirect_dispatch_buffer)?;
        let indirect_draw = StageBuffer::wrap(cmd_pool, &indirect_draw_data, indirect_draw_buffer)?;

        let indirect = Indirect::new(super_indirect_buffer, &indirect_dispatch, &indirect_draw);
        let player_event_uniform = HostBuffer::new(cmd_pool.device(), &[PlayerEvent::nothing()])?;

        let max_subgroup_size = cmd_pool.device().get_max_subgroup_size();
        println!("MAX subgroup size={}",max_subgroup_size);
        let mut specialization_constants = SpecializationConstants::new();
        specialization_constants.entry_uint(1,max_subgroup_size);//GROUP_SIZE
        specialization_constants.entry_uint(2,cap.world_size.width() as u32);//CHUNKS_X
        specialization_constants.entry_uint(3,cap.world_size.depth() as u32);//CHUNKS_Z
        specialization_constants.entry_uint(4,cap.faces_to_be_inserted_chunk_capacity);//
        specialization_constants.entry_uint(5,cap.faces_to_be_removed_chunk_capacity);//
        specialization_constants.entry_uint(6,2);//BROAD_PHASE_CELL_SIZE
        specialization_constants.entry_uint(7,8);//BROAD_PHASE_CELL_CAPACITY
        specialization_constants.entry_float(100,0.7);//BLOCK_COLLISION_FRICTION
        specialization_constants.entry_float(101,0.01);//BLOCK_COLLISION_MINIMUM_BOUNCE
        specialization_constants.entry_float(102,1.0);//PHYSICS_SIMULATION_DELTA_TIME_PER_STEP
        specialization_constants.entry_float(103,0.01);//BONE_COLLISION_FORCE_PER_AREA_UNIT
        specialization_constants.entry_float(104,0.2);//IMPULSE_AVERAGING_OVER_TIMESETP
        specialization_constants.entry_float(105,0.001f32);//GRAVITY
        specialization_constants.entry_float(106,0.99);//DAMPING_COEFFICIENT

        specialization_constants.entry_uint(300,cap.max_bones as u32);//MAX_BONES
        specialization_constants.entry_uint(301,cap.max_sensors as u32);//MAX_SENSORS
        specialization_constants.entry_uint(302,cap.max_faces as u32);//MAX_FACES
        specialization_constants.entry_uint(303,cap.max_rand_uint as u32);//MAX_RAND_UINT
        // specialization_constants.entry_uint(304,cap.max_neural_net_layers as u32);//MAX_NEURAL_NET_LAYERS
        specialization_constants.entry_uint(305,cap.max_tmp_faces_copy as u32);//MAX_TMP_FACES_COPY
        // specialization_constants.entry_uint(306,cap.max_world_block_meta as u32);//MAX_WORLD_BLOCKS_TO_UPDATE
        specialization_constants.entry_uint(307,cap.max_blocks_to_be_inserted_or_removed as u32);//MAX_BLOCKS_TO_BE_INSERTED_OR_REMOVED
        specialization_constants.entry_uint(308,cap.max_faces_to_be_inserted as u32);//MAX_FACES_TO_BE_INSERTED
        specialization_constants.entry_uint(309,cap.max_faces_to_be_removed as u32);//MAX_FACES_TO_BE_REMOVED
        specialization_constants.entry_uint(310,cap.max_htm_entities as u32);//MAX_HTM_ENTITIES
        specialization_constants.entry_uint(311,cap.max_ann_entities as u32);//MAX_ANN_ENTITIES
        specialization_constants.entry_uint(312,cap.max_particles as u32);//MAX_PARTICLES

        specialization_constants.entry_uint(400, super::world_generation::SEA_LEVEL);
        specialization_constants.entry_float(401, super::world_generation::FREEZING_TEMPERATURE);
        specialization_constants.entry_float(402, super::world_generation::SWAMP_HUMIDITY);
        specialization_constants.entry_float(403, super::world_generation::DESERT_HUMIDITY);
        specialization_constants.entry_float(404, super::world_generation::LARGE_SCALE);
        specialization_constants.entry_float(405, super::world_generation::CHUNK_SCALE);
        specialization_constants.entry_float(406, super::world_generation::TEMPERATURE_SCALE);
        specialization_constants.entry_float(407, super::world_generation::HUMIDITY_SCALE);
        specialization_constants.entry_float(408, super::world_generation::RESOURCE_TYPE_SCALE);
        specialization_constants.entry_float(409, super::world_generation::HAS_RESOURCE_SCALE);
        specialization_constants.entry_uint(410, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_millis() as u32);//RAND_SEED
        Ok(Self {
            cap,
            specialization_constants,
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            rand_uint:rand_uint_buffer,
            faces:face_buffer,
            world:world_buffer,
            sampler,
            player_event_uniform,
            htm_entities_buffer,
            ann_entities_buffer,
            particles,
            collision_grid: grid_buffer,
            global_mutables,
            indirect_dispatch,
            indirect_draw,
            tmp_faces_copy,
            blocks_to_be_inserted_or_removed: blocks_to_be_inserted_or_removed_buffer,
            faces_to_be_inserted: faces_to_be_inserted_buffer,
            faces_to_be_removed: faces_to_be_removed_buffer,
            indirect,
            bones:bones_buffer,
            default_global_mutables:mutables,
        })
    }
    pub fn build(self) -> Result<Foundations, Error> {
        let Self {
            cap,
            rand_uint,
            specialization_constants,
            face_count_per_chunk_buffer,
            htm_entities_buffer ,
            ann_entities_buffer,
            opaque_and_transparent_face_buffer,
            faces,
            faces_to_be_inserted,
            faces_to_be_removed,
            tmp_faces_copy,
            blocks_to_be_inserted_or_removed,
            player_event_uniform,
            indirect_dispatch,
            indirect_draw,
            world,
            bones,
            particles,
            collision_grid,
            global_mutables,
            indirect,
            sampler,
            default_global_mutables
        } = self;
        let global_mutables = global_mutables.take()?.take_gpu();
        let _ = indirect_dispatch.take()?.take_gpu();
        let _ = indirect_draw.take()?.take_gpu();
        let tmp_faces_copy = tmp_faces_copy.take()?;
        Ok(Foundations {
            rand_uint,
            specialization_constants,
            faces_to_be_inserted,
            faces_to_be_removed,
            tmp_faces_copy,
            blocks_to_be_inserted_or_removed,
            player_event_uniform,
            face_count_per_chunk_buffer,
            opaque_and_transparent_face_buffer,
            faces,
            particles,
            world,
            cap,
            bones,
            collision_grid,
            global_mutables,
            indirect,
            sampler,
            htm_entities_buffer ,
            ann_entities_buffer,
            default_global_mutables
        })
    }
}

pub struct Foundations {
    htm_entities_buffer: SubBuffer<HtmEntity, Storage>,
    ann_entities_buffer: SubBuffer<AnnEntity, Storage>,
    specialization_constants: SpecializationConstants,
    faces: SubBuffer<Face, Storage>,
    rand_uint: SubBuffer<u32, Storage>,
    face_count_per_chunk_buffer: SubBuffer<Face, Storage>,
    opaque_and_transparent_face_buffer: SubBuffer<Face, Storage>,
    faces_to_be_inserted: SubBuffer<Face, Storage>,
    faces_to_be_removed: SubBuffer<u32, Storage>,
    cap: FoundationsCapacity,
    tmp_faces_copy: SubBuffer<u32, Storage>,
    blocks_to_be_inserted_or_removed: SubBuffer<u32, Storage>,
    player_event_uniform: HostBuffer<PlayerEvent, Uniform>,
    world: SubBuffer<Block, Storage>,
    bones: SubBuffer<Bone, Storage>,
    particles: SubBuffer<Particle, Storage>,
    global_mutables: SubBuffer<GlobalMutables, Storage>,
    collision_grid: SubBuffer<u32, Storage>,
    indirect: Indirect,
    sampler: Sampler,
    default_global_mutables:GlobalMutables,
}

impl Foundations {
    pub fn htm_entities_buffer(&self)-> &SubBuffer<HtmEntity, Storage>{
        &self.htm_entities_buffer
    }
    pub fn ann_entities_buffer(&self)-> &SubBuffer<AnnEntity, Storage>{
        &self.ann_entities_buffer
    }
    pub fn specialization_constants(&self) -> &SpecializationConstants{
        &self.specialization_constants
    }
    pub fn tmp_faces_copy(&self) -> &SubBuffer<u32, Storage> {
        &self.tmp_faces_copy
    }
    pub fn faces(&self) -> &SubBuffer<Face, Storage> {
        &self.faces
    }
    pub fn face_count_per_chunk_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.face_count_per_chunk_buffer
    }
    pub fn opaque_and_transparent_face_buffer(&self) -> &SubBuffer<Face, Storage> {
        &self.opaque_and_transparent_face_buffer
    }
    pub fn player_event_uniform(&self) -> &HostBuffer<PlayerEvent, Uniform> {
        &self.player_event_uniform
    }
    pub fn player_event_uniform_mut(&mut self) -> &mut HostBuffer<PlayerEvent, Uniform> {
        &mut self.player_event_uniform
    }
    pub fn cap(&self) -> &FoundationsCapacity {
        &self.cap
    }
    pub fn world_size(&self) -> &WorldSize {
        &self.cap.world_size
    }
    pub fn indirect(&self) -> &Indirect {
        &self.indirect
    }
    pub fn bones(&self) -> &SubBuffer<Bone, Storage> {
        &self.bones
    }
    pub fn faces_to_be_inserted(&self) -> &SubBuffer<Face, Storage> {
        &self.faces_to_be_inserted
    }
    pub fn faces_to_be_removed(&self) -> &SubBuffer<u32, Storage> {
        &self.faces_to_be_removed
    }
    pub fn blocks_to_be_inserted_or_removed(&self) -> &SubBuffer<u32, Storage> {
        &self.blocks_to_be_inserted_or_removed
    }
    pub fn particles(&self) -> &SubBuffer<Particle, Storage> {
        &self.particles
    }
    pub fn rand_uint(&self) -> &SubBuffer<u32, Storage> {
        &self.rand_uint
    }
    pub fn world(&self) -> &SubBuffer<Block, Storage> {
        &self.world
    }
    pub fn default_global_mutables(&self) -> &GlobalMutables {
        &self.default_global_mutables
    }
    pub fn global_mutables(&self) -> &SubBuffer<GlobalMutables, Storage> {
        &self.global_mutables
    }
    pub fn collision_grid(&self) -> &SubBuffer<u32, Storage> {
        &self.collision_grid
    }
}
