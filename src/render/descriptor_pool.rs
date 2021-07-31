use ash::vk;
use crate::render::swap_chain::SwapChain;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::buffer::{Buffer, Uniform, Gpu, Storage};
use crate::render::data::VertexSource;
use crate::render::sampler::Sampler;
use crate::render::imageview::{ImageView, Color};
use ash::vk::{DescriptorImageInfo, DescriptorBufferInfo};
use crate::render::host_buffer::HostBuffer;
use crate::render::compute::ComputePipeline;

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
    pub fn new(descriptor_layout:&DescriptorLayout, swapchain: &SwapChain) -> Result<Self, ash::vk::Result> {
        Self::manual_new(descriptor_layout, swapchain.len() as u32, swapchain.device())
    }
    pub fn new_for_compute(descriptor_layout:&DescriptorLayout, compute: &ComputePipeline) -> Result<Self, ash::vk::Result> {
        Self::manual_new(descriptor_layout, 1, compute.device())
    }
    pub fn manual_new(descriptor_layout:&DescriptorLayout, size:u32, device:&Device) -> Result<Self, ash::vk::Result> {
        let pool_sizes = descriptor_layout.manual_pool_sizes(size);

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(size)
            .pool_sizes(&pool_sizes);

        unsafe {
            device.inner().create_descriptor_pool(&descriptor_pool_create_info, None)
        }.map(|raw|Self{raw,device:device.clone()})
    }

    pub fn create_sets_with_same_layout(&self, layout:DescriptorLayout, num:usize)->Result<Vec<DescriptorSet>,vk::Result>{
        let layouts:Vec<DescriptorLayout> = std::iter::repeat(layout).take(num).collect();
        self.create_sets(&layouts)
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
    pub  fn update_uniform_buffer<T:Copy>(&self,binding:u32,buffer:&HostBuffer<T,Uniform>){
        unsafe { self.update_uniform_buffer_raw(binding,&buffer.buffer().descriptor_info()) }
    }
    pub unsafe fn update_uniform_buffer_raw(&self,binding:u32,descriptor_info:&DescriptorBufferInfo){
        unsafe { self.update_buffer_raw(binding,descriptor_info, vk::DescriptorType::UNIFORM_BUFFER) }
    }
    pub fn update_storage_buffer<T:Copy>(&self,binding:u32,buffer:&Buffer<T,Storage>) {
        unsafe { self.update_storage_buffer_raw(binding,&buffer.descriptor_info()) }
    }
    pub unsafe fn update_storage_buffer_raw(&self,binding:u32,descriptor_info:&DescriptorBufferInfo) {
        unsafe { self.update_buffer_raw(binding,descriptor_info, vk::DescriptorType::STORAGE_BUFFER) }
    }
    pub unsafe fn update_buffer_raw(&self,binding:u32,descriptor_info:&DescriptorBufferInfo,descriptor_type:vk::DescriptorType){
        assert_eq!(self.layout.layout(binding).descriptor_type, descriptor_type, "Tried to bind buffer to {} ",binding);

        let descriptor_write_sets = vk::WriteDescriptorSet::builder()
            .dst_set(self.raw())
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(descriptor_type)
            .buffer_info(std::slice::from_ref(descriptor_info));

        unsafe {
            self.device.inner().update_descriptor_sets(std::slice::from_ref(&descriptor_write_sets), &[]);
        }
    }
    pub fn update_sampler(&self,binding:u32,sampler:&Sampler, image_view:&ImageView<Color>) {
        unsafe { self.update_sampler_raw(binding, &sampler.descriptor_info(image_view)); }
    }
    pub unsafe fn update_sampler_raw(&self,binding:u32,descriptor_info:&DescriptorImageInfo){
        assert_eq!(self.layout.layout(binding).descriptor_type, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);

        let descriptor_write_sets = vk::WriteDescriptorSet::builder()
            .dst_set(self.raw)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(descriptor_info));

        unsafe {
            self.device.inner().update_descriptor_sets(std::slice::from_ref(&descriptor_write_sets), &[]);
        }
    }
}

