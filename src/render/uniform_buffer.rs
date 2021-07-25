use crate::render::buffer::{Buffer, Uniform};
use crate::render::device::Device;
use crate::render::data::VertexSource;
use crate::render::descriptor_layout::DescriptorLayout;
use std::fmt::{Debug, Formatter};
use crate::render::swap_chain::SwapChain;
use ash::vk;
use std::ops::{Deref, DerefMut};

pub struct UniformBuffers<T:Copy, const size: usize> {
    pub data: [T; size],
    buff: Vec<Buffer<T, Uniform>>,
}

impl <T:Copy, const size: usize>  Deref for UniformBuffers<T, size>{
    type Target = [T; size];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl <T:Copy, const size: usize>  DerefMut for UniformBuffers<T, size>{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
impl<T:Copy, const size: usize> UniformBuffers<T, size> {
    pub fn new_array(device:&Device, swapchain:&SwapChain, data: [T; size]) -> Result<Self, ash::vk::Result> {
        let buff:Result<Vec<Buffer<T, Uniform>>,vk::Result> = (0..swapchain.len()).into_iter().map(|_|Buffer::<T,Uniform>::with_capacity(device, size)).collect();
        buff.map(|buff|Self{buff,data})
    }
    pub fn buffers(&self)->std::slice::Iter<Buffer<T, Uniform>>{
        self.buff.iter()
    }
    pub fn map_unmap(&mut self, frame_idx:usize, f:impl FnOnce(&mut [T])) -> Result<(), ash::vk::Result> {
        self.buff[frame_idx].map_unmap(0,size,f)
    }
    pub fn device(&self)->&Device{
        self.device()
    }
    pub fn flush(&mut self, frame_idx:usize) -> Result<(), ash::vk::Result> {
        let Self{ data, buff } = self;
        buff[frame_idx].map_copy_unmap(0,&*data)
    }
    pub fn create_descriptor_layout(&self) -> Result<DescriptorLayout, ash::vk::Result> {
        DescriptorLayout::new_uniform(self)
    }
    pub fn buffer(&self, frame_idx:usize) -> & Buffer<T, Uniform>{
        &self.buff[frame_idx]
    }
}

impl <V:Copy+Debug, const size:usize> Debug for UniformBuffers<V,size>{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl<T:VertexSource> UniformBuffers<T, 1> {
    pub fn new(device:&Device, swapchain:&SwapChain, data: T) -> Result<Self, ash::vk::Result> {
        Self::new_array(device, swapchain,[data])
    }
}

