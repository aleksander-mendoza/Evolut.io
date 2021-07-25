use ash::vk;
use crate::render::swap_chain::SwapChain;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::buffer::{Buffer, Uniform};
use crate::render::data::VertexSource;
use crate::render::sampler::Sampler;
use crate::render::imageview::{ImageView, Color};

pub struct DescriptorPool{
    raw:vk::DescriptorPool,
    device:Device
}
impl Drop for DescriptorPool{
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_descriptor_pool(self.raw, None)}
    }
}

impl DescriptorPool{
    pub fn device(&self)->&Device{
        &self.device
    }
    pub fn new(device: &Device, descriptor_layout:&DescriptorLayout, swapchain: &SwapChain) -> Result<Self, ash::vk::Result> {
        let pool_sizes = descriptor_layout.pool_sizes(swapchain.len());

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(swapchain.len() as u32)
            .pool_sizes(&pool_sizes);

        unsafe {
            device.inner().create_descriptor_pool(&descriptor_pool_create_info, None)
        }.map(|raw|Self{raw,device:device.clone()})
    }

    pub fn create_sets(&self, layouts:&[DescriptorLayout])->Result<Vec<DescriptorSet>,vk::Result>{
        let raw_layouts:Vec<vk::DescriptorSetLayout> = layouts.iter().map(|r|r.raw()).collect();
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.raw)
            .set_layouts(&raw_layouts);

        let descriptor_sets = unsafe { self.device().inner().allocate_descriptor_sets(&descriptor_set_allocate_info) }?;
        Ok(descriptor_sets.into_iter().zip(layouts.iter()).map(|(raw,layout)|DescriptorSet{raw,layout:layout.clone(),device:self.device().clone()}).collect())
    }
}

pub struct DescriptorSet{
    raw:vk::DescriptorSet,
    layout: DescriptorLayout,
    device:Device
}

impl DescriptorSet{

    pub fn raw(&self)->vk::DescriptorSet{
        self.raw
    }

    pub fn update_buffer<T:Copy>(&self,binding:u32,buffer:&Buffer<T,Uniform>){
        assert_eq!(self.layout.layout(binding).descriptor_type, vk::DescriptorType::UNIFORM_BUFFER, "Tried to bind buffer to {} ",binding);
        let descriptor_info = buffer.descriptor_info();

        let descriptor_write_sets = vk::WriteDescriptorSet::builder()
            .dst_set(self.raw)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&descriptor_info));

        unsafe {
            self.device.inner().update_descriptor_sets(std::slice::from_ref(&descriptor_write_sets), &[]);
        }
    }

    pub fn update_sampler(&self,binding:u32,sampler:&Sampler, image_view:&ImageView<Color>){
        assert_eq!(self.layout.layout(binding).descriptor_type, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        let descriptor_info = sampler.descriptor_info(image_view);

        let descriptor_write_sets = vk::WriteDescriptorSet::builder()
            .dst_set(self.raw)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&descriptor_info));

        unsafe {
            self.device.inner().update_descriptor_sets(std::slice::from_ref(&descriptor_write_sets), &[]);
        }
    }
}

