use ash::vk;
use crate::render::data::VertexSource;
use crate::render::device::Device;
use ash::version::{DeviceV1_0, InstanceV1_0};
use crate::render::instance::Instance;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::marker::PhantomData;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use crate::render::fence::Fence;
use crate::render::submitter::Submitter;
use std::fmt::Debug;
use std::ptr::NonNull;
use std::slice::{Iter, IterMut};

pub trait Type {
    const SHARING_MODE: vk::SharingMode;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags;
    const USAGE: vk::BufferUsageFlags;
}

pub trait CpuWriteable: Type {}

pub trait GpuWriteable: Type {}

pub struct Uniform {}

impl Type for Uniform {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw());
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
}


impl CpuWriteable for Uniform {}

pub struct Gpu {}

impl Type for Gpu {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::VERTEX_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}

impl GpuWriteable for Gpu {}

pub struct GpuIndirect {}

impl Type for GpuIndirect {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}

impl GpuWriteable for GpuIndirect {}

pub struct Cpu {}

impl Type for Cpu {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw());
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;
}

impl CpuWriteable for Cpu {}

// pub struct RefBuffer<'a, V, T:Type> {
//     r:&'a Buffer<V,T>,
//     offset:usize,
//     len:usize
// }
//
// impl <'a, V, T:Type> Deref for RefBuffer<'a, V, T>{
//     type Target = Buffer<V,T>;
//
//     fn deref(&self) -> &Self::Target {
//         self.r
//     }
// }

pub struct Buffer<V: Copy, T: Type> {
    capacity: usize,
    raw: vk::Buffer,
    mem: vk::DeviceMemory,
    device: Device,
    _u: PhantomData<T>,
    _v: PhantomData<V>,
}

impl<V: Copy, T: Type> Drop for Buffer<V, T> {
    fn drop(&mut self) {
        unsafe {
            self.device.inner().destroy_buffer(self.raw, None);
            self.device.inner().free_memory(self.mem, None);
        }
    }
}

impl<V: Copy, T: Type> Buffer<V, T> {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn offset(&self) -> usize {
        0
    }
    pub fn len(&self) -> usize {
        self.capacity()
    }
    pub fn mem_capacity(&self) -> vk::DeviceSize {
        (std::mem::size_of::<V>() * self.capacity) as u64
    }
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

impl<V: Copy> Buffer<V, Uniform> {
    pub fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.raw(),
            offset: 0,
            range: self.mem_capacity(),
        }
    }
}

impl<V: Copy, T: CpuWriteable> Buffer<V, T> {
    pub fn new(device: &Device, data: &[V]) -> Result<Self, vk::Result> {
        let mut slf = Self::with_capacity(device, data.len())?;
        slf.map_copy_unmap(0, data);
        Ok(slf)
    }

    pub fn map_copy_unmap(&mut self, offset: usize, data: &[V]) -> Result<(), vk::Result> {
        unsafe {
            self.unsafe_map_unmap(offset, data.len(), |ptr| ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()))
        }
    }
    pub fn map_unmap(&mut self, offset: usize, len: usize, f: impl FnOnce(&mut [V])) -> Result<(), vk::Result> {
        unsafe {
            self.unsafe_map_unmap(offset, len, |ptr| f(std::slice::from_raw_parts_mut(ptr, len)))
        }
    }
    unsafe fn map(&mut self, offset: usize, len: usize) -> Result<*mut V, vk::Result> {
        assert!(offset + len <= self.capacity);
        self.device.inner().map_memory(
            self.mem,
            offset as u64,
            len as u64,
            vk::MemoryMapFlags::empty(),
        ).map(|v| v as *mut V)
    }
    unsafe fn unmap(&mut self) {
        self.device.inner().unmap_memory(self.mem)
    }
    unsafe fn unsafe_map_unmap(&mut self, offset: usize, len: usize, f: impl FnOnce(*mut V)) -> Result<(), vk::Result> {
        f(self.map(offset, len)?);
        self.unmap();
        Ok(())
    }
}

pub struct Vector<V: Copy, C: CpuWriteable> {
    cpu: Buffer<V, C>,
    len: usize,
    data_ptr: NonNull<V>,
}
impl<V: Copy, C: CpuWriteable> Vector<V, C> {

    pub fn buffer(&self) -> &Buffer<V, C> {
        &self.cpu
    }
    pub fn capacity(&self) -> usize {
        self.cpu.capacity()
    }
    pub fn device(&self) -> &Device {
        self.cpu.device()
    }
    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self, vk::Result> {
        let mut cpu = Buffer::with_capacity(device, capacity)?;
        let data_ptr = unsafe { NonNull::new_unchecked(cpu.map(0, capacity)?) };
        Ok(Self { cpu, data_ptr, len: 0 })
    }
    pub unsafe fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity());
        self.len = len;
    }
    pub fn as_slice_mut(&mut self) -> &mut [V] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr.as_ptr(), self.len) }
    }
    pub fn as_slice(&self) -> &[V] {
        unsafe { std::slice::from_raw_parts(self.data_ptr.as_ptr(), self.len) }
    }
    pub fn new(device: &Device, data: &[V]) -> Result<Self, vk::Result> {
        let mut slf = Self::with_capacity(device, data.len())?;
        unsafe { slf.set_len(data.len()) }
        slf.as_slice_mut().copy_from_slice(data);
        Ok(slf)
    }
    pub fn reallocate(&mut self, new_capacity: usize) -> Result<(), vk::Result> {
        let mut cpu = Buffer::<V, C>::with_capacity(self.device(), new_capacity)?;
        let data_ptr = unsafe { NonNull::new_unchecked(cpu.map(0, new_capacity)?) };
        self.len = self.len.min(new_capacity);
        unsafe {
            data_ptr.as_ptr().copy_from_nonoverlapping(self.data_ptr.as_ptr(), self.len)
        }
        self.cpu = cpu;
        self.data_ptr = data_ptr;
        Ok(())
    }
    unsafe fn unsafe_push(&mut self, v: V) {
        self.data_ptr.as_ptr().offset(self.len() as isize).write(v)
    }
    pub fn swap_remove(&mut self, idx:usize) -> V{
        let last = self.len()-1;
        self.swap(idx,last);
        unsafe{self.set_len(last)}
        unsafe{self.data_ptr.as_ptr().offset(last as isize).read()}
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
    pub fn iter(&self) -> Iter<'_, V> {
        self.cpu.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<'_, V> {
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

    pub fn new(device: &Device, cmd: &CommandPool, data: &[V]) -> Result<Submitter<Self>, vk::Result> {
        let mut slf = Submitter::new(Self::with_capacity(device, data.len())?,cmd)?;
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
    pub fn new_vertex_buffer(device: &Device, cmd: &CommandPool, data: &[V]) -> Result<Submitter<Self>, vk::Result> {
        Self::new(device, cmd, data)
    }
}

pub type IndirectBuffer = StageBuffer<vk::DrawIndirectCommand, Cpu, GpuIndirect>;

impl IndirectBuffer {
    pub fn new_indirect_buffer(device: &Device, cmd: &CommandPool, data: &[vk::DrawIndirectCommand]) -> Result<Submitter<Self>, vk::Result> {
        Self::new(device, cmd, data)
    }
}