use ash::vk;
use crate::render::submitter::Submitter;
use crate::render::buffer::{GpuWriteable, CpuWriteable, Cpu, GpuIndirect, Buffer, Gpu};
use std::ops::{Index, IndexMut};
use crate::render::device::Device;
use crate::render::command_pool::CommandPool;
use crate::render::data::VertexSource;
use crate::render::vector::Vector;
use std::collections::btree_map::IterMut;

pub struct StageBuffer<V: Copy, C: CpuWriteable, G: GpuWriteable> {
    gpu: Buffer<V, G>,
    cpu: Vector<V, C>,
    has_unflushed_changes:bool
}

impl<V: Copy, C: CpuWriteable, G: GpuWriteable> StageBuffer<V, C, G> {
    pub fn len(&self) -> usize {
        self.cpu.len()
    }
    pub fn capacity(&self) -> usize {
        self.cpu.capacity()
    }
    pub fn device(&self) -> &Device {
        self.cpu.device()
    }
    pub fn cpu(&self) -> &Buffer<V, C> {
        self.cpu.buffer()
    }
    /**Returns true if the backing buffer need to be reallocated. In such cases the GPU memory becomes invalidated, and you need to re-record all command buffers that make use of it.
    Whether any reallocation occurred or not, the GPU is never flushed automatically. You need to decide when the most optimal time for flush is*/
    pub fn push(&mut self, v: V) -> Result<bool, vk::Result> {
        let out = Ok(if self.len() == self.capacity() {
            self.reallocate(16.max(self.capacity() * 2))?;
            unsafe { self.cpu.unsafe_push(v) }
            true
        } else {
            unsafe { self.cpu.unsafe_push(v) }
            false
        });
        self.has_unflushed_changes = true;
        out
    }
    pub fn is_empty(&self) -> bool {
        self.cpu.is_empty()
    }
    pub fn has_unflushed_changes(&self) -> bool {
        self.has_unflushed_changes
    }
    pub fn mark_with_unflushed_changes(&mut self) {
        self.has_unflushed_changes = true
    }
    pub fn mark_with_no_changes(&mut self) {
        self.has_unflushed_changes = false
    }
    pub fn pop(&mut self) -> V {
        self.has_unflushed_changes = true;
        self.cpu.pop()
    }
    pub unsafe fn set_len(&mut self, len: usize) {
        if len != self.len(){
            self.cpu.set_len(len);
            self.has_unflushed_changes = true;
        }
    }
    pub fn swap(&mut self, idx1:usize, idx2:usize) {
        self.cpu.swap(idx1,idx2);
        self.has_unflushed_changes = true;
    }
    pub fn swap_remove(&mut self, idx:usize) -> V {
        self.has_unflushed_changes = true;
        self.cpu.swap_remove(idx)
    }
    pub fn as_slice_mut(&mut self) -> &mut [V] {
        self.cpu.as_slice_mut()
    }
    pub fn as_slice(&self) -> &[V] {
        self.cpu.as_slice()
    }
    pub fn iter(&self) -> std::slice::Iter<V> {
        self.cpu.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<V> {
        self.cpu.iter_mut()
    }
    /**The GPU memory becomes invalidated and needs to be flushed again manually. You also need to re-record all command buffers that make use of it.*/
    pub fn reallocate(&mut self, new_capacity: usize) -> Result<(), vk::Result> {
        self.cpu.reallocate(new_capacity)?;
        self.gpu = Buffer::with_capacity(self.device(), new_capacity)?;
        self.has_unflushed_changes = true;
        Ok(())
    }
    pub fn gpu(&self) -> &Buffer<V, G> {
        &self.gpu
    }
    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self, vk::Result> {
        let cpu = Vector::with_capacity(device, capacity)?;
        let gpu = Buffer::with_capacity(device, capacity)?;
        Ok(Self { cpu, gpu, has_unflushed_changes: false })
    }

    pub fn new(cmd: &CommandPool, data: &[V]) -> Result<Submitter<Self>, vk::Result> {
        let mut slf = Submitter::new(Self::with_capacity(cmd.device(), data.len())?,cmd)?;
        unsafe { slf.set_len(data.len()) }
        slf.as_slice_mut().copy_from_slice(data);
        slf.flush_to_gpu()?;
        Ok(slf)
    }
}

impl <V:Copy,C:CpuWriteable,G:GpuWriteable> Submitter<StageBuffer<V,C,G>>{
    pub fn flush_to_gpu(&mut self) -> Result<(), vk::Result> {
        let (cmd,buff) = self.inner_val();
        cmd.reset()?
            .reset()?
            .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?
            .copy_from_staged_if_has_changes(buff)
            .end()?;
        self.inner_mut().submit()
    }
}

impl<V: Copy, C: CpuWriteable, G: GpuWriteable> Index<usize> for StageBuffer<V, C, G> {
    type Output = V;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cpu[index]
    }
}

impl<V: Copy, C: CpuWriteable, G: GpuWriteable> IndexMut<usize> for StageBuffer<V, C, G> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cpu[index]
    }
}
pub type VertexBuffer<V: VertexSource> = StageBuffer<V, Cpu, Gpu>;

impl<V: Copy> VertexBuffer<V> {
    pub fn new_vertex_buffer(cmd: &CommandPool, data: &[V]) -> Result<Submitter<Self>, vk::Result> {
        Self::new(cmd, data)
    }
}

pub type IndirectBuffer = StageBuffer<vk::DrawIndirectCommand, Cpu, GpuIndirect>;

impl IndirectBuffer {
    pub fn new_indirect_buffer(cmd: &CommandPool, data: &[vk::DrawIndirectCommand]) -> Result<Submitter<Self>, vk::Result> {
        Self::new(cmd, data)
    }
}