use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use crate::buffer::{Uniform, Buffer};
use std::rc::Rc;
use crate::data::VertexSource;

struct DescriptorLayoutInner{
    raw:vk::DescriptorSetLayout,
    device: Device,
    binding: u32
}
impl Drop for DescriptorLayoutInner{
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_descriptor_set_layout(self.raw, None) }

    }
}
#[derive(Clone)]
pub struct DescriptorLayout{
    inner:Rc<DescriptorLayoutInner>
}

impl DescriptorLayout{
    pub fn raw(&self)->vk::DescriptorSetLayout{
        self.inner.raw
    }
    pub fn binding(&self) -> u32 {
        self.inner.binding
    }
    pub fn from_uniform<T:VertexSource>(binding:u32, buffer:&Buffer<T,Uniform>) -> Result<DescriptorLayout, ash::vk::Result> {
        Self::new(buffer.device(),vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: buffer.capacity() as u32,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        })
    }

    pub fn new(device: &Device, layout:vk::DescriptorSetLayoutBinding) -> Result<Self, ash::vk::Result> {
        let ubo_layout_bindings = [layout];

        let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&ubo_layout_bindings);
        unsafe {
            device.inner().create_descriptor_set_layout(&ubo_layout_create_info, None)
        }.map(|raw|Self{inner:Rc::new(DescriptorLayoutInner{raw,binding:layout.binding, device:device.clone()})})
    }
}

