


use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::shader_module::{ShaderModule, Compute};


use ash::vk;


use failure::Error;






use crate::render::compute::{ComputePipeline, UniformBufferBinding, ComputeDescriptorsBuilder};


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

pub struct PhysicsResources {
    broad_phase_collision_detection: ShaderModule<Compute>,
    broad_phase_collision_detection_cleanup: ShaderModule<Compute>,
    narrow_phase_collision_detection: ShaderModule<Compute>,
    update_bones: ShaderModule<Compute>,
    update_particles: ShaderModule<Compute>,
    update_ann_entities: ShaderModule<Compute>,
    // update_entity_lidars: ShaderModule<Compute>,
}




impl PhysicsResources {
    pub fn new(cmd_pool: &CommandPool, _foundations:&FoundationInitializer) -> Result<Self, failure::Error> {
        let broad_phase_collision_detection = ShaderModule::new(include_glsl!("assets/shaders/broad_phase_collision_detection.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let broad_phase_collision_detection_cleanup = ShaderModule::new(include_glsl!("assets/shaders/broad_phase_collision_detection_cleanup.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let narrow_phase_collision_detection = ShaderModule::new(include_glsl!("assets/shaders/narrow_phase_collision_detection.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_particles = ShaderModule::new(include_glsl!("assets/shaders/update_particles.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_bones = ShaderModule::new(include_glsl!("assets/shaders/update_bones.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_ann_entities = ShaderModule::new(include_glsl!("assets/shaders/update_ann_entities.comp", kind: comp, target: vulkan1_1) as &[u32], cmd_pool.device())?;
        // let update_htm_entities = ShaderModule::new(include_glsl!("assets/shaders/update_htm_entities.comp", kind: comp) as &[u32], cmd_pool.device())?;
        Ok(Self {
            broad_phase_collision_detection,
            broad_phase_collision_detection_cleanup,
            narrow_phase_collision_detection,
            update_particles,
            update_bones,
            update_ann_entities
        })
    }
}
impl ComputeResources for PhysicsResources{
    type Compute = Physics;

    fn make_computable(self, cmd_pool: &CommandPool, foundations:&Foundations) -> Result<Physics, Error> {
        let Self {
            broad_phase_collision_detection,
            broad_phase_collision_detection_cleanup,
            update_particles,
            narrow_phase_collision_detection,
            update_bones,
            update_ann_entities,
            // feed_forward_net,
        } = self;
        let mut descriptors = ComputeDescriptorsBuilder::new();
        let uniform_binding = descriptors.uniform_buffer(foundations.player_event_uniform().buffer());//0
        descriptors.storage_buffer(foundations.global_mutables());//1
        descriptors.storage_buffer(foundations.collision_grid());//2
        descriptors.storage_buffer(foundations.particles());//4
        descriptors.storage_buffer(foundations.indirect().super_buffer());//5
        descriptors.storage_buffer(foundations.bones());//6
        descriptors.storage_buffer(foundations.world());//7
        descriptors.storage_buffer(foundations.faces());//8
        descriptors.storage_buffer(foundations.htm_entities_buffer());//9
        descriptors.storage_buffer(foundations.ann_entities_buffer());//10
        descriptors.storage_buffer(foundations.rand_uint());//11
        let descriptors = descriptors.build(cmd_pool.device())?;
        // descriptors.storage_buffer(foundations.block_properties());
        // descriptors.storage_buffer(foundations.particles());
        // descriptors.storage_buffer(foundations.sensors());

        // descriptors.storage_buffer(foundations.constraints());
        // descriptors.storage_buffer(foundations.muscles());
        let sc = foundations.specialization_constants().build();
        let broad_phase_collision_detection = descriptors.build("main", broad_phase_collision_detection,&sc)?;
        let broad_phase_collision_detection_cleanup = descriptors.build("main", broad_phase_collision_detection_cleanup,&sc)?;
        let update_bones = descriptors.build("main", update_bones,&sc)?;
        let update_ann_entities = descriptors.build("main", update_ann_entities,&sc)?;
        // let feed_forward_net = descriptors.build("main", feed_forward_net)?;
        let update_particles = descriptors.build("main", update_particles,&sc)?;
        let narrow_phase_collision_detection = descriptors.build("main", narrow_phase_collision_detection,&sc)?;
        Ok(Physics {
            narrow_phase_collision_detection,
            update_particles,
            update_ann_entities,
            // feed_forward_net,
            broad_phase_collision_detection,
            broad_phase_collision_detection_cleanup,
            update_bones,
            uniform_binding,
        })
    }
}


pub struct Physics {
    uniform_binding: UniformBufferBinding<PlayerEvent>,
    broad_phase_collision_detection: ComputePipeline,
    broad_phase_collision_detection_cleanup: ComputePipeline,
    narrow_phase_collision_detection: ComputePipeline,
    update_bones: ComputePipeline,
    update_particles: ComputePipeline,
    update_ann_entities: ComputePipeline,
    // feed_forward_net: ComputePipeline,
}

impl Computable for Physics {
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer,foundations:&Foundations) -> Result<(), Error> {
        cmd
            .bind_compute_descriptors(&self.broad_phase_collision_detection_cleanup)
            .bind_compute_pipeline(&self.broad_phase_collision_detection_cleanup)
            .dispatch_indirect(foundations.indirect().broad_phase_collision_detection_cleanup(), 0)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.collision_grid())
            ])
            .bind_compute_pipeline(&self.broad_phase_collision_detection)
            .dispatch_indirect(foundations.indirect().broad_phase_collision_detection(), 0)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.collision_grid())
            ])
            .bind_compute_pipeline(&self.narrow_phase_collision_detection)
            .dispatch_indirect(foundations.indirect().narrow_phase_collision_detection(), 0)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.bones())
            ])
            .bind_compute_pipeline(&self.update_ann_entities)
            .dispatch_indirect(foundations.indirect().update_ann_entities(), 0)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.bones())
            ])
            .bind_compute_pipeline(&self.update_bones)
            .dispatch_indirect(foundations.indirect().update_bones(), 0)

            // .bind_compute_pipeline(&self.update_particles)
            // .dispatch_indirect(foundations.indirect().update_particles(), 0)
            // .bind_compute_pipeline(&self.update_entity_lidars)
            // .dispatch_indirect(foundations.indirect().update_entity_lidars(), 0)


            // .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
            //     make_shader_buffer_barrier(foundations.particles()),
            //     make_shader_buffer_barrier(foundations.constraints())
            // ])
            // .bind_compute_pipeline(&self.solve_constraints)
            // .dispatch_indirect(foundations.indirect().solve_constraints(), 0)
            // .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
            //     make_shader_buffer_barrier(foundations.particles()),
            // ])

            // .bind_compute_pipeline(&self.agent_sensory_inputs)
            // .dispatch_indirect(foundations.indirect().agent_sensory_input_update(), 0)
            // .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
            //     make_shader_buffer_barrier(foundations.persistent_floats()),
            // ])
            // .bind_compute_pipeline(&self.feed_forward_net)
            // .dispatch_indirect(foundations.indirect().feed_forward_net(), 0)
        ;
        Ok(())
    }

    fn update_uniforms(&mut self, player: &mut Player,foundations:&mut Foundations) {
        if let Some(event) = player.pop_event(){
            foundations.player_event_uniform_mut().as_slice_mut()[0] = event;
        }else {
            foundations.player_event_uniform_mut().as_slice_mut()[0].make_nothing();
        }
    }

}