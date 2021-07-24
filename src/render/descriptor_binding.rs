use ash::vk;
use crate::render::buffer::{Buffer, Uniform};
use crate::render::data::VertexSource;
use ash::vk::DescriptorSetLayoutBinding;
use crate::render::sampler::Sampler;
use crate::render::uniform_buffer::UniformBuffers;

pub trait DescriptorBinding{
    fn create_binding(&self, binding:u32)->vk::DescriptorSetLayoutBinding;
}

impl <V,const size:usize> DescriptorBinding for UniformBuffers<V,size>{
    fn create_binding(&self, binding: u32) -> DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size as u32,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        }
    }
}

impl DescriptorBinding for Sampler{
    fn create_binding(&self, binding: u32) -> DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
        }
    }
}