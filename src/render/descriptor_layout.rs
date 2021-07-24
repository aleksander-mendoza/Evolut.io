use ash::vk;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use crate::render::buffer::{Uniform, Buffer};
use std::rc::Rc;
use crate::render::data::VertexSource;
use crate::render::descriptor_binding::DescriptorBinding;
use std::ops::Deref;
use crate::render::sampler::Sampler;
use ash::vk::DescriptorPoolSize;

struct DescriptorLayoutInner{
    raw:vk::DescriptorSetLayout,
    device: Device,
    bindings_to_layout: Vec<Option<vk::DescriptorSetLayoutBinding>>
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
    pub fn pool_sizes(&self, swapchain_images:usize) -> Vec<DescriptorPoolSize> {
        let mut sizes = Vec::new();
        for layout in &self.inner.bindings_to_layout{
            if let Some(layout) = layout{
                sizes.push(vk::DescriptorPoolSize {
                    ty: layout.descriptor_type,
                    descriptor_count: swapchain_images as u32,
                });
            }
        }
        sizes
    }
    pub fn layout(&self, binding:u32)->&vk::DescriptorSetLayoutBinding{
        self.inner.bindings_to_layout[binding as usize].as_ref().unwrap()
    }

    pub fn raw(&self)->vk::DescriptorSetLayout{
        self.inner.raw
    }
    pub fn new_uniform<T:VertexSource>(buffer:&Buffer<T,Uniform>) -> Result<DescriptorLayout, ash::vk::Result> {
        DescriptorLayout::new(buffer.device(),&[buffer.create_binding(0)])
    }
    pub fn new_sampler_uniform<T:VertexSource>(sampler:&Sampler, buffer:&Buffer<T,Uniform>) -> Result<DescriptorLayout, ash::vk::Result> {
        DescriptorLayout::new(buffer.device(),&[sampler.create_binding(0),buffer.create_binding(1)])
    }
    pub fn new(device: &Device, layouts:&[vk::DescriptorSetLayoutBinding]) -> Result<Self, ash::vk::Result> {
        let max_binding = layouts.iter().map(|l|l.binding).max().expect("No descriptor set layout bindings provided!") as usize;
        let mut bindings_to_layout = vec![None;max_binding+1];
        for layout in layouts{
            let  prev = &mut bindings_to_layout[layout.binding as usize];
            assert!(prev.is_none());
            prev.insert(layout.clone());
        }
        let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&layouts);
        unsafe {
            device.inner().create_descriptor_set_layout(&ubo_layout_create_info, None)
        }.map(|raw|Self{inner:Rc::new(DescriptorLayoutInner{raw,bindings_to_layout, device:device.clone()})})
    }
}


