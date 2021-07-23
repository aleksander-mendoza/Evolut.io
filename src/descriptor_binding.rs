use ash::vk;
use crate::buffer::{Buffer, Uniform};
use crate::data::VertexSource;
use ash::vk::DescriptorSetLayoutBinding;
use crate::sampler::Sampler;

pub trait DescriptorBinding{
    fn create_binding(&self, binding:u32)->vk::DescriptorSetLayoutBinding;
}

impl <V:VertexSource> DescriptorBinding for Buffer<V,Uniform>{
    fn create_binding(&self, binding: u32) -> DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: self.capacity() as u32,
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