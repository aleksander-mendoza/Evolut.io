use ash::vk;
use crate::render::data::VertexSource;
use crate::render::device::Device;
use ash::version::{DeviceV1_0, InstanceV1_0};
use crate::render::instance::Instance;
use std::ops::{Deref, DerefMut, Index, IndexMut, RangeFrom, RangeBounds};
use std::marker::PhantomData;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::fence::Fence;
use crate::render::submitter::Submitter;
use std::fmt::Debug;
use std::ptr::NonNull;
use std::slice::{Iter, IterMut};
use crate::render::buffer_type::{AsDescriptor, CpuWriteable, BufferType};
use std::rc::Rc;
use std::collections::Bound;
use crate::render::buffer::Buffer;


pub struct OwnedBuffer<V: Copy, T: BufferType> {
    capacity: usize,
    raw: vk::Buffer,
    mem: vk::DeviceMemory,
    device: Device,
    _u: PhantomData<T>,
    _v: PhantomData<V>,
}

impl<V: Copy, T: BufferType> Drop for OwnedBuffer<V, T> {
    fn drop(&mut self) {
        unsafe {
            self.device.inner().destroy_buffer(self.raw, None);
            self.device.inner().free_memory(self.mem, None);
        }
    }
}

impl <V: Copy, T: BufferType> Buffer<V, T> for OwnedBuffer<V, T>{
    fn device(&self) -> &Device {
        &self.device
    }
    fn raw(&self) -> vk::Buffer {
        self.raw
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn offset(&self) -> usize {
        0
    }
    fn len(&self) -> usize {
        self.capacity()
    }
    fn raw_mem(&self) -> vk::DeviceMemory{
        self.mem
    }
}


impl<V: Copy, T: BufferType> OwnedBuffer<V, T> {

    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self, vk::Result> {
        let vertex_buffer_create_info = vk::BufferCreateInfo::builder()
            .size((std::mem::size_of::<V>() * capacity) as u64)
            .usage(T::USAGE)
            .sharing_mode(T::SHARING_MODE);

        let vertex_buffer = unsafe { device.inner().create_buffer(&vertex_buffer_create_info, None) }?;

        let mem_requirements = unsafe { device.inner().get_buffer_memory_requirements(vertex_buffer) };
        let memory_type = device.find_memory_type(mem_requirements, T::REQUIRED_MEMORY_FLAGS);

        let allocate_info = vk::MemoryAllocateInfo::builder()
            .memory_type_index(memory_type)
            .allocation_size(mem_requirements.size);

        let vertex_buffer_memory = unsafe { device.inner().allocate_memory(&allocate_info, None) }?;
        unsafe {
            device.inner().bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0)?;
        }
        Ok(Self {
            capacity,
            raw: vertex_buffer,
            mem: vertex_buffer_memory,
            device: device.clone(),
            _u: PhantomData,
            _v: PhantomData,
        })
    }
}



impl<V: Copy, T: CpuWriteable> OwnedBuffer<V, T> {
    pub fn new(device: &Device, data: &[V]) -> Result<Self, vk::Result> {
        let mut slf = Self::with_capacity(device, data.len())?;
        slf.map_copy_unmap(0, data);
        Ok(slf)
    }
    pub fn map_copy_unmap(&mut self, offset: usize, data: &[V]) -> Result<(), vk::Result> {
        super::buffer::map_copy_unmap(self, offset, data)
    }
    pub fn map_unmap(&mut self, offset: usize, len: usize, f: impl FnOnce(&mut [V])) -> Result<(), vk::Result> {
        super::buffer::map_unmap(self, offset, len, f)
    }

}
