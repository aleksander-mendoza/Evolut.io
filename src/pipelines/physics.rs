


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
    update_bones: ShaderModule<Compute>,
    update_particles: ShaderModule<Compute>,
    particle_uniform: HostBuffer<PlayerEvent, Uniform>,
    feed_forward_net: ShaderModule<Compute>,
    agent_sensory_inputs: ShaderModule<Compute>,
}




impl PhysicsResources {
    pub fn new(cmd_pool: &CommandPool, _foundations:&FoundationInitializer) -> Result<Self, failure::Error> {
        // let generate_faces = ShaderModule::new(include_glsl!("assets/shaders/generate_faces.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let broad_phase_collision_detection = ShaderModule::new(include_glsl!("assets/shaders/broad_phase_collision_detection.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let broad_phase_collision_detection_cleanup = ShaderModule::new(include_glsl!("assets/shaders/broad_phase_collision_detection_cleanup.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_particles = ShaderModule::new(include_glsl!("assets/shaders/update_particles.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_bones = ShaderModule::new(include_glsl!("assets/shaders/update_bones.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let feed_forward_net = ShaderModule::new(include_glsl!("assets/shaders/feed_forward_net.comp", kind: comp, target: vulkan1_1) as &[u32], cmd_pool.device())?;
        let agent_sensory_inputs = ShaderModule::new(include_glsl!("assets/shaders/agent_sensory_input_update.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let particle_uniform = HostBuffer::new(cmd_pool.device(), &[PlayerEvent::nothing()])?;
        Ok(Self { broad_phase_collision_detection, broad_phase_collision_detection_cleanup, particle_uniform, update_particles, update_bones, agent_sensory_inputs, feed_forward_net })
    }
}
impl ComputeResources for PhysicsResources{
    type Compute = Physics;

    fn make_computable(self, cmd_pool: &CommandPool, foundations:&Foundations) -> Result<Physics, Error> {
        let Self {
            broad_phase_collision_detection,
            broad_phase_collision_detection_cleanup,
            update_particles,
            particle_uniform,
            update_bones,
            agent_sensory_inputs,
            feed_forward_net,
        } = self;
        let mut descriptors = ComputeDescriptorsBuilder::new();
        let uniform_binding = descriptors.uniform_buffer(particle_uniform.buffer());
        descriptors.storage_buffer(foundations.constants());
        descriptors.storage_buffer(foundations.particles());
        descriptors.storage_buffer(foundations.neural_net_layers());
        descriptors.storage_buffer(foundations.persistent_floats());
        descriptors.storage_buffer(foundations.indirect().super_buffer());
        descriptors.storage_buffer(foundations.bones());
        descriptors.storage_buffer(foundations.world());
        descriptors.storage_buffer(foundations.faces());
        descriptors.storage_buffer(foundations.block_properties());
        descriptors.storage_buffer(foundations.collision_grid());
        // descriptors.storage_buffer(foundations.sensors());

        // descriptors.storage_buffer(foundations.constraints());
        // descriptors.storage_buffer(foundations.muscles());
        let descriptors = descriptors.build(cmd_pool.device())?;
        let broad_phase_collision_detection = descriptors.build("main", broad_phase_collision_detection)?;
        let broad_phase_collision_detection_cleanup = descriptors.build("main", broad_phase_collision_detection_cleanup)?;
        let update_bones = descriptors.build("main", update_bones)?;
        let agent_sensory_inputs = descriptors.build("main", agent_sensory_inputs)?;
        let feed_forward_net = descriptors.build("main", feed_forward_net)?;
        let update_particles = descriptors.build("main", update_particles)?;
        Ok(Physics {
            update_particles,
            agent_sensory_inputs,
            feed_forward_net,
            broad_phase_collision_detection,
            broad_phase_collision_detection_cleanup,
            update_bones,
            uniform_binding,
            player_event_uniform: particle_uniform,
        })
    }
}


pub struct Physics {
    //The order matters because of drop!
    uniform_binding: UniformBufferBinding<PlayerEvent>,
    broad_phase_collision_detection: ComputePipeline,
    broad_phase_collision_detection_cleanup: ComputePipeline,
    update_bones: ComputePipeline,
    update_particles: ComputePipeline,
    agent_sensory_inputs: ComputePipeline,
    feed_forward_net: ComputePipeline,
    player_event_uniform: HostBuffer<PlayerEvent, Uniform>,
}

impl Computable for Physics {
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer,foundations:&Foundations) -> Result<(), Error> {
        cmd
            .bind_compute_descriptors(&self.update_particles)
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
            .bind_compute_pipeline(&self.update_bones)
            .dispatch_indirect(foundations.indirect().update_bones(), 0)
            .bind_compute_pipeline(&self.update_particles)
            .dispatch_indirect(foundations.indirect().update_particles(), 0)


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

    fn update_uniforms(&mut self, player: &mut Player) {
        if let Some(event) = player.pop_event(){
            self.player_event_uniform.as_slice_mut()[0] = event;
        }else {
            self.player_event_uniform.as_slice_mut()[0].make_nothing();
        }
    }

}