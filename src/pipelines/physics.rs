


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

pub struct PhysicsResources {
    collision_detection: ShaderModule<Compute>,
    solve_constraints: ShaderModule<Compute>,
    update_bones: ShaderModule<Compute>,
    particle_uniform: HostBuffer<ThrowUniform, Uniform>,
}



#[derive(Copy, Clone, Debug)]
#[repr(C, align(16))]
pub struct ThrowUniform {
    position: Vec3,
    velocity: Vec3,
}

impl PhysicsResources {
    pub fn new(cmd_pool: &CommandPool, _foundations:&FoundationInitializer) -> Result<Self, failure::Error> {
        let collision_detection = ShaderModule::new(include_glsl!("assets/shaders/collision_detection.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let solve_constraints = ShaderModule::new(include_glsl!("assets/shaders/solve_constraints.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let update_bones = ShaderModule::new(include_glsl!("assets/shaders/bones.comp", kind: comp) as &[u32], cmd_pool.device())?;
        let particle_uniform = HostBuffer::new(cmd_pool.device(), &[ThrowUniform {
            position: Vec3(glm::vec3(0., 0., 0.)),
            velocity: Vec3(glm::vec3(0., 0., 0.)),
        }])?;
        Ok(Self { collision_detection, solve_constraints, particle_uniform, update_bones })
    }
}
impl ComputeResources for PhysicsResources{
    type Compute = Physics;

    fn make_computable(self, cmd_pool: &CommandPool, foundations:&Foundations) -> Result<Physics, Error> {
        let Self {
            collision_detection,
            solve_constraints,
            particle_uniform,
            update_bones
        } = self;
        let mut descriptors = ComputeDescriptorsBuilder::new();
        let uniform_binding = descriptors.uniform_buffer(particle_uniform.buffer());
        descriptors.storage_buffer(foundations.constants());
        descriptors.storage_buffer(foundations.particles());
        descriptors.storage_buffer(foundations.collision_grid());
        descriptors.storage_buffer(foundations.constraints());
        descriptors.storage_buffer(foundations.indirect_dispatch());
        descriptors.storage_buffer(foundations.bones());
        let descriptors = descriptors.build(cmd_pool.device())?;
        let collision_detection = descriptors.build("main", collision_detection)?;
        let solve_constraints = descriptors.build("main", solve_constraints)?;
        let update_bones = descriptors.build("main", update_bones)?;
        Ok(Physics {
            collision_detection,
            solve_constraints,
            update_bones,
            uniform_binding,
            particle_uniform,
        })
    }
}


pub struct Physics {
    //The order matters because of drop!
    uniform_binding: UniformBufferBinding<ThrowUniform>,
    collision_detection: ComputePipeline,
    solve_constraints: ComputePipeline,
    update_bones: ComputePipeline,
    particle_uniform: HostBuffer<ThrowUniform, Uniform>,
}

impl Computable for Physics {
    fn record_compute_cmd_buffer(&self, cmd: &mut CommandBuffer,foundations:&Foundations) -> Result<(), Error> {
        cmd.bind_compute_pipeline(&self.collision_detection)
            .bind_compute_descriptors(&self.collision_detection)
            .dispatch_indirect(foundations.indirect().collision_detection(), 0)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.particles()),
                make_shader_buffer_barrier(foundations.constraints())
            ])
            .bind_compute_pipeline(&self.solve_constraints)
            .dispatch_indirect(foundations.indirect().solve_constraints(), 0)
            .buffer_barriers(vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, &[
                make_shader_buffer_barrier(foundations.particles()),
            ])
            .bind_compute_pipeline(&self.update_bones)
            .dispatch_indirect(foundations.indirect().update_bones(), 0)
        ;
        Ok(())
    }

    fn update_uniforms(&mut self, player: &Player) {
        let throw_uniform = &mut self.particle_uniform.as_slice_mut()[0];
        throw_uniform.position.0 = player.location().clone();
        throw_uniform.velocity.0 = player.throw_velocity().clone();
    }

}