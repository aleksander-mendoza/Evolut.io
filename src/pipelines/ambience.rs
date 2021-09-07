use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::{ShaderModule, Compute};
use ash::vk;
use failure::Error;
use crate::render::compute::{ComputePipeline, UniformBufferBinding, ComputeDescriptorsBuilder, ComputeDescriptors};
use crate::render::host_buffer::HostBuffer;
use crate::pipelines::player::Player;
use crate::render::uniform_types::Vec3;
use crate::render::buffer_type::{Storage, Cpu, Uniform, GpuIndirect};
use crate::render::buffer::{make_shader_buffer_barrier, Buffer};
use crate::pipelines::foundations::{Foundations, FoundationInitializer};
use crate::pipelines::computable::{ComputeResources, Computable};
use crate::render::submitter::Submitter;
use crate::blocks::world_size::{CHUNK_VOLUME_IN_CELLS, CHUNK_VOLUME};
use crate::pipelines::player_event::{PlayerEvent, EventType};

pub struct AmbienceResources {
    update_player_events: ShaderModule<Compute>,
    update_ambience: ShaderModule<Compute>,
    update_ambience_faces: ShaderModule<Compute>,
    update_ambience_flush_insertions: ShaderModule<Compute>,
    update_ambience_prepare_face_offsets: ShaderModule<Compute>,
    update_ambience_prepare_insertions: ShaderModule<Compute>,
}


impl AmbienceResources {
    pub fn new(cmd_pool: &CommandPool, _foundations: &FoundationInitializer) -> Result<Self, failure::Error> {
        let update_player_events = ShaderModule::new(include_glsl!("assets/shaders/update_player_events.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_ambience = ShaderModule::new(include_glsl!("assets/shaders/update_ambience.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_ambience_faces = ShaderModule::new(include_glsl!("assets/shaders/update_ambience_faces.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_ambience_flush_insertions = ShaderModule::new(include_glsl!("assets/shaders/update_ambience_flush_insertions.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_ambience_prepare_face_offsets = ShaderModule::new(include_glsl!("assets/shaders/update_ambience_prepare_face_offsets.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_ambience_prepare_insertions = ShaderModule::new(include_glsl!("assets/shaders/update_ambience_prepare_insertions.comp", kind: comp) as &[u32], cmd_pool.device())?;
        Ok(Self {
            update_player_events,
            update_ambience,
            update_ambience_faces,
            update_ambience_flush_insertions,
            update_ambience_prepare_face_offsets,
            update_ambience_prepare_insertions,
        })
    }
}

impl ComputeResources for AmbienceResources {
    type Compute = Ambience;

    fn make_computable(self, cmd_pool: &CommandPool, foundations: &Foundations) -> Result<Ambience, Error> {
        let Self {
            update_player_events,
            update_ambience,
            update_ambience_faces,
            update_ambience_flush_insertions,
            update_ambience_prepare_face_offsets,
            update_ambience_prepare_insertions,
        } = self;
        let mut descriptors = ComputeDescriptorsBuilder::new();
        let uniform_binding = descriptors.uniform_buffer(foundations.player_event_uniform().buffer());
        descriptors.storage_buffer(foundations.global_mutables());
        descriptors.storage_buffer(foundations.world_blocks_to_update());
        descriptors.storage_buffer(foundations.faces_to_be_inserted());
        descriptors.storage_buffer(foundations.faces_to_be_removed());
        descriptors.storage_buffer(foundations.indirect().super_buffer());
        descriptors.storage_buffer(foundations.tmp_faces_copy());
        descriptors.storage_buffer(foundations.world());
        descriptors.storage_buffer(foundations.faces());
        descriptors.storage_buffer(foundations.world_copy());
        descriptors.storage_buffer(foundations.blocks_to_be_inserted_or_removed());
        let descriptors = descriptors.build(cmd_pool.device())?;

        let sc = foundations.specialization_constants().build();
        let update_player_events = descriptors.build("main", update_player_events, &sc)?;
        let update_ambience = descriptors.build("main", update_ambience, &sc)?;
        let update_ambience_faces = descriptors.build("main", update_ambience_faces,&sc)?;
        let update_ambience_flush_insertions = descriptors.build("main", update_ambience_flush_insertions,&sc)?;
        let update_ambience_prepare_face_offsets = descriptors.build("main", update_ambience_prepare_face_offsets,&sc)?;
        let update_ambience_prepare_insertions = descriptors.build("main", update_ambience_prepare_insertions,&sc)?;
        Ok(Ambience {
            update_player_events,
            update_ambience,
            update_ambience_faces,
            update_ambience_flush_insertions,
            update_ambience_prepare_face_offsets,
            update_ambience_prepare_insertions,
        })
    }
}


pub struct Ambience {
    update_player_events: ComputePipeline,
    update_ambience: ComputePipeline,
    update_ambience_faces: ComputePipeline,
    update_ambience_flush_insertions: ComputePipeline,
    update_ambience_prepare_face_offsets: ComputePipeline,
    update_ambience_prepare_insertions: ComputePipeline,
}

impl Computable for Ambience {
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer, foundations: &Foundations) -> Result<(), Error> {
        cmd
            .bind_compute_descriptors(&self.update_player_events)
            .bind_compute_pipeline(&self.update_player_events)
            .dispatch_1d(1)
            .bind_compute_pipeline(&self.update_ambience)
            .dispatch_1d(1)
        ;
        Ok(())
    }

    fn update_uniforms(&mut self, player: &mut Player,foundations:&mut Foundations) {}
}