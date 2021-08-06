use crate::render::owned_buffer::{OwnedBuffer};
use std::ptr::NonNull;
use crate::render::device::Device;
use ash::vk;
use std::ops::{Deref, DerefMut};
use crate::render::buffer_type::CpuWriteable;
use crate::render::buffer::{Buffer, map};

pub struct HostBuffer<V: Copy, C: CpuWriteable> {
    cpu: OwnedBuffer<V, C>,
    data_ptr: NonNull<V>,
}
impl<V: Copy, C: CpuWriteable> HostBuffer<V, C> {

    pub fn buffer(&self) -> &OwnedBuffer<V, C> {
        &self.cpu
    }
    pub fn capacity(&self) -> usize {
        self.cpu.capacity()
    }
    pub fn device(&self) -> &Device {
        self.cpu.device()
    }
    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self, vk::Result> {
        let mut cpu = OwnedBuffer::with_capacity(device, capacity)?;
        let data_ptr = unsafe { NonNull::new_unchecked(map(&mut cpu,0, capacity)?) };
        Ok(Self { cpu, data_ptr })
    }
    pub fn as_slice_mut(&mut self) -> &mut [V] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr.as_ptr(), self.capacity()) }
    }
    pub fn as_slice(&self) -> &[V] {
        unsafe { std::slice::from_raw_parts(self.data_ptr.as_ptr(), self.capacity()) }
    }
    pub fn new(device: &Device, data: &[V]) -> Result<Self, vk::Result> {
        let mut slf = Self::with_capacity(device, data.len())?;
        slf.as_slice_mut().copy_from_slice(data);
        Ok(slf)
    }
}

impl<V: Copy, C: CpuWriteable> Deref for HostBuffer<V, C> {
    type Target = [V];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<V: Copy, C: CpuWriteable> DerefMut for HostBuffer<V, C> {

    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}
