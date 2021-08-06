use crate::render::owned_buffer::{OwnedBuffer};
use std::ptr::NonNull;
use crate::render::device::Device;
use ash::vk;
use std::ops::{Deref, DerefMut};
use crate::render::host_buffer::HostBuffer;
use crate::render::buffer_type::CpuWriteable;


pub struct Vector<V: Copy, C: CpuWriteable> {
    cpu: HostBuffer<V, C>,
    len: vk::DeviceSize,
}
impl<V: Copy, C: CpuWriteable> Vector<V, C> {

    pub fn buffer(&self) -> &OwnedBuffer<V, C> {
        &self.cpu.buffer()
    }
    pub fn capacity(&self) -> vk::DeviceSize {
        self.cpu.elements()
    }
    pub fn len(&self) -> vk::DeviceSize {
        self.len
    }
    pub fn device(&self) -> &Device {
        self.cpu.device()
    }
    pub fn with_capacity(device: &Device, max_elements: vk::DeviceSize) -> Result<Self, vk::Result> {
        let mut cpu = HostBuffer::with_capacity(device, max_elements)?;
        Ok(Self { cpu, len: 0 })
    }
    pub unsafe fn set_len(&mut self, len: vk::DeviceSize) {
        assert!(len <= self.capacity());
        self.len = len;
    }
    pub fn as_slice_mut(&mut self) -> &mut [V] {
        debug_assert!(self.len() <= self.capacity());
        unsafe { std::slice::from_raw_parts_mut(self.cpu.as_mut_ptr(), self.len() as usize) }
    }
    pub fn as_slice(&self) -> &[V] {
        debug_assert!(self.len() <= self.capacity());
        unsafe { std::slice::from_raw_parts(self.cpu.as_ptr(), self.len() as usize) }
    }
    pub fn new(device: &Device, data: &[V]) -> Result<Self, vk::Result> {
        let mut slf = Self::with_capacity(device, data.len() as u64)?;
        unsafe { slf.set_len(data.len() as u64) }
        slf.as_slice_mut().copy_from_slice(data);
        Ok(slf)
    }
    pub fn reallocate(&mut self, new_max_elements: vk::DeviceSize) -> Result<(), vk::Result> {
        let mut cpu = HostBuffer::<V, C>::with_capacity(self.device(), new_max_elements)?;
        self.len = self.len.min(new_max_elements);
        unsafe {
            cpu.as_mut_ptr().copy_from_nonoverlapping(self.as_ptr(), self.len() as usize)
        }
        self.cpu = cpu;
        Ok(())
    }
    pub unsafe fn unsafe_push(&mut self, v: V) {
        self.cpu.as_mut_ptr().offset(self.len() as isize).write(v);
        self.len+=1
    }
    pub fn swap_remove(&mut self, idx:usize) -> V{
        let last = self.len()-1;
        self.swap(idx,last as usize);
        unsafe{self.set_len(last)}
        unsafe{self.as_ptr().offset(last as isize).read()}
    }
    pub fn push(&mut self, v: V) -> Result<bool, vk::Result> {
        Ok(if self.len() == self.capacity() {
            self.reallocate(16.max(self.capacity() * 2))?;
            unsafe { self.unsafe_push(v) }
            true
        } else {
            unsafe { self.unsafe_push(v) }
            false
        })
    }
    pub fn pop(&mut self) -> V {
        let v = *self.as_slice().last().unwrap();
        self.len -= 1;
        v
    }
}

impl<V: Copy, C: CpuWriteable> Deref for Vector<V, C> {
    type Target = [V];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<V: Copy, C: CpuWriteable> DerefMut for Vector<V, C> {

    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}