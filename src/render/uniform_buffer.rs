use crate::render::buffer::{Buffer, Uniform};
use crate::render::device::Device;
use crate::render::data::VertexSource;
use crate::render::descriptor_layout::DescriptorLayout;

pub struct UniformBuffer<T:VertexSource, const size: usize> {
    data: [T; size],
    buff: Buffer<T, Uniform>,
}

impl<T:VertexSource, const size: usize> UniformBuffer<T, size> {
    pub fn new_array(device:&Device, data: [T; size]) -> Result<Self, ash::vk::Result> {
        Buffer::<T,Uniform>::new(device, &data).map(|buff|Self{
            data,
            buff
        })
    }
    pub fn device(&self)->&Device{
        self.device()
    }
    pub fn create_descriptor_layout(&self) -> Result<DescriptorLayout, ash::vk::Result> {
        DescriptorLayout::new_uniform(&self.buff)
    }
    pub fn buffer(&self) -> & Buffer<T, Uniform>{
        &self.buff
    }
}


impl<T:VertexSource> UniformBuffer<T, 1> {
    pub fn new(device:&Device, data: T) -> Result<Self, ash::vk::Result> {
        Self::new_array(device, [data])
    }
}

