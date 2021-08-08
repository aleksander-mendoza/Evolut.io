use ash::vk;
use crate::render::submitter::Submitter;
use crate::render::owned_buffer::{OwnedBuffer};
use std::ops::{Index, IndexMut};
use crate::render::device::Device;
use crate::render::command_pool::CommandPool;

use crate::render::vector::Vector;

use crate::render::buffer_type::{GpuWriteable, CpuWriteable, GpuIndirect, Cpu, Gpu};
use crate::render::buffer::Buffer;
use crate::render::subbuffer::SubBuffer;
use std::marker::PhantomData;

pub type StageOwnedBuffer<V, C, G> = StageBuffer<V,C,G,OwnedBuffer<V,G>>;
pub type StageSubBuffer<V, C, G> = StageBuffer<V,C,G,SubBuffer<V,G>>;
pub struct StageBuffer<V: Copy, C: CpuWriteable, G: GpuWriteable, B:Buffer<V, G>> {
    gpu: B,
    cpu: Vector<V, C>,
    has_unflushed_changes:bool,
    _g:PhantomData<G>
}

impl<V: Copy, C: CpuWriteable, G: GpuWriteable, B:Buffer<V, G>> StageBuffer<V, C, G, B> {
    pub fn len(&self) -> vk::DeviceSize {
        self.cpu.len()
    }
    pub fn capacity(&self) -> vk::DeviceSize {
        self.cpu.capacity()
    }
    pub fn device(&self) -> &Device {
        self.cpu.device()
    }
    pub fn cpu(&self) -> &OwnedBuffer<V, C> {
        self.cpu.buffer()
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


    pub fn swap(&mut self, idx1:usize, idx2:usize) {
        self.cpu.swap(idx1,idx2);
        self.has_unflushed_changes = true;
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

    pub fn gpu(&self) -> &B {
        &self.gpu
    }
    pub fn take_gpu(self) -> B {
        let Self{gpu, ..} = self;
        gpu
    }
    pub fn with_capacity(device: &Device, max_elements: vk::DeviceSize) -> Result<Self, vk::Result> {
        let cpu = Vector::with_capacity(device, max_elements)?;
        let gpu = B::with_capacity(device, max_elements)?;
        Ok(Self { cpu, gpu, has_unflushed_changes: false,_g:PhantomData })
    }
    pub fn new(cmd: &CommandPool, data: &[V]) -> Result<Submitter<Self>, vk::Result> {
        Self::new_with_capacity(cmd,data,data.len() as u64)
    }
    pub fn new_with_capacity(cmd: &CommandPool, data: &[V], max_elements:vk::DeviceSize) -> Result<Submitter<Self>, vk::Result> {
        assert!(max_elements>=data.len() as u64);
        let mut slf = Submitter::new(Self::with_capacity(cmd.device(), max_elements)?,cmd)?;
        unsafe { slf.set_len(data.len() as u64) }
        slf.as_slice_mut().copy_from_slice(data);
        slf.flush_to_gpu()?;
        Ok(slf)
    }
    pub unsafe fn set_len(&mut self, len: vk::DeviceSize) {
        if len != self.len(){
            self.cpu.set_len(len);
            self.has_unflushed_changes = true;
        }
    }
}
impl<V: Copy, C: CpuWriteable, G: GpuWriteable> StageBuffer<V, C, G, SubBuffer<V,G>> {
    pub fn wrap(cmd: &CommandPool, data: &[V], gpu:SubBuffer<V,G>)->Result<Submitter<Self>, vk::Result>{
        assert!((data.len()*std::mem::size_of::<V>()) as u64 <= gpu.bytes(), "len={}, size_of={}, bytes={}", data.len(), std::mem::size_of::<V>(), gpu.bytes() );
        let cpu = Vector::with_capacity(cmd.device(), data.len() as u64)?;
        let mut slf = Submitter::new(Self{cpu,gpu,has_unflushed_changes:false, _g:PhantomData},cmd)?;
        unsafe { slf.set_len(data.len() as u64) }
        slf.as_slice_mut().copy_from_slice(data);
        slf.flush_to_gpu()?;
        Ok(slf)
    }
}

impl<V: Copy, C: CpuWriteable, G: GpuWriteable> StageBuffer<V, C, G, OwnedBuffer<V,G>> {
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
    pub fn pop(&mut self) -> V {
        self.has_unflushed_changes = true;
        self.cpu.pop()
    }
    pub fn swap_remove(&mut self, idx:usize) -> V {
        self.has_unflushed_changes = true;
        self.cpu.swap_remove(idx)
    }
    /**The GPU memory becomes invalidated and needs to be flushed again manually. You also need to re-record all command buffers that make use of it.*/
    pub fn reallocate(&mut self, new_max_elements: vk::DeviceSize) -> Result<(), vk::Result> {
        self.cpu.reallocate(new_max_elements)?;
        self.gpu = OwnedBuffer::with_capacity(self.device(), new_max_elements)?;
        self.has_unflushed_changes = true;
        Ok(())
    }

}

impl <V:Copy,C:CpuWriteable,G:GpuWriteable,B:Buffer<V, G>> Submitter<StageBuffer<V,C,G,B>>{
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

impl<V: Copy, C: CpuWriteable, G: GpuWriteable,B:Buffer<V, G>> Index<usize> for StageBuffer<V, C, G,B> {
    type Output = V;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cpu[index]
    }
}

impl<V: Copy, C: CpuWriteable, G: GpuWriteable,B:Buffer<V, G>> IndexMut<usize> for StageBuffer<V, C, G,B> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cpu[index]
    }
}
pub type VertexBuffer<V,B> = StageBuffer<V, Cpu, Gpu,B>;
pub type VertexOwnedBuffer<V> = VertexBuffer<V,OwnedBuffer<V,Gpu>>;
pub type VertexSubBuffer<V> = VertexBuffer<V,SubBuffer<V,Gpu>>;

impl<V: Copy,B:Buffer<V, Gpu>> VertexBuffer<V,B> {
    pub fn new_vertex_buffer(cmd: &CommandPool, data: &[V]) -> Result<Submitter<Self>, vk::Result> {
        Self::new(cmd, data)
    }
}

pub type IndirectBuffer<B> = StageBuffer<vk::DrawIndirectCommand, Cpu, GpuIndirect,B>;
pub type IndirectOwnedBuffer = IndirectBuffer<OwnedBuffer<vk::DrawIndirectCommand, GpuIndirect>>;
pub type IndirectSubBuffer = IndirectBuffer<SubBuffer<vk::DrawIndirectCommand, GpuIndirect>>;
impl <B:Buffer<vk::DrawIndirectCommand, GpuIndirect>> IndirectBuffer<B> {
    pub fn new_indirect_buffer(cmd: &CommandPool, data: &[vk::DrawIndirectCommand]) -> Result<Submitter<Self>, vk::Result> {
        Self::new(cmd, data)
    }
}


pub type IndirectDispatchBuffer<B> = StageBuffer<vk::DispatchIndirectCommand, Cpu, GpuIndirect, B>;
pub type IndirectDispatchOwnedBuffer = IndirectDispatchBuffer<OwnedBuffer<vk::DispatchIndirectCommand, GpuIndirect>>;
pub type IndirectDispatchSubBuffer = IndirectDispatchBuffer<SubBuffer<vk::DispatchIndirectCommand, GpuIndirect>>;
impl <B:Buffer<vk::DispatchIndirectCommand, GpuIndirect>> IndirectDispatchBuffer<B> {
    pub fn new_indirect_dispatch_buffer(cmd: &CommandPool, data: &[vk::DispatchIndirectCommand]) -> Result<Submitter<Self>, vk::Result> {
        Self::new(cmd, data)
    }
}