use ash::vk;
use crate::render::data::VertexSource;
use crate::render::device::Device;
use ash::version::{DeviceV1_0, InstanceV1_0};
use crate::render::instance::Instance;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use crate::render::command_pool::CommandPool;
use crate::render::fence::Fence;
use crate::render::gpu_future::GpuFuture;
use std::fmt::Debug;

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

pub struct Buffer<V, T: Type> {
    capacity: usize,
    raw: vk::Buffer,
    mem: vk::DeviceMemory,
    device: Device,
    _u: PhantomData<T>,
    _v: PhantomData<V>,
}

impl<V, T: Type> Drop for Buffer<V, T> {
    fn drop(&mut self) {
        unsafe {
            self.device.inner().destroy_buffer(self.raw, None);
            self.device.inner().free_memory(self.mem, None);
        }
    }
}

impl<V, T: Type> Buffer<V, T> {
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

impl<V> Buffer<V, Uniform> {
    pub fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.raw(),
            offset: 0,
            range: self.mem_capacity(),
        }
    }
}

impl<V, T: CpuWriteable> Buffer<V, T> {
    pub fn new(device: &Device, data: &[V]) -> Result<Self, vk::Result> {
        let mut slf = Self::with_capacity(device, data.len())?;
        slf.map_copy_unmap(0, data);
        Ok(slf)
    }

    pub fn map_copy_unmap(&mut self, offset: usize, data: &[V]) -> Result<(), vk::Result> {
        unsafe {
            self.unsafe_map_unmap(offset, data.len(), |ptr|  ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()))
        }
    }
    pub fn map_unmap(&mut self, offset: usize, len: usize, f: impl FnOnce(&mut [V])) -> Result<(), vk::Result> {
        unsafe {
            self.unsafe_map_unmap(offset, len, |ptr| f(std::slice::from_raw_parts_mut(ptr, len)))
        }
    }
    unsafe fn unsafe_map_unmap(&mut self, offset: usize, len: usize, f: impl FnOnce(*mut V)) -> Result<(), vk::Result> {
        assert!(offset + len <= self.capacity);
        let data_ptr = self.device.inner().map_memory(
            self.mem,
            offset as u64,
            len as u64,
            vk::MemoryMapFlags::empty(),
        )? as *mut V;
        f(data_ptr);
        self.device.inner().unmap_memory(self.mem);
        Ok(())
    }
}


pub struct StageBuffer<V, C: CpuWriteable, G: GpuWriteable> {
    cpu: Buffer<V, C>,
    gpu: Buffer<V, G>,
}

impl<V: Debug, C: CpuWriteable, G: GpuWriteable> StageBuffer<V, C, G> {
    pub fn cpu(&self) -> &Buffer<V, C> {
        &self.cpu
    }
    pub fn gpu(&self) -> &Buffer<V, G> {
        &self.gpu
    }
    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self, vk::Result> {
        let cpu = Buffer::with_capacity(device, capacity)?;
        let gpu = Buffer::with_capacity(device, capacity)?;
        Ok(Self { cpu, gpu })
    }

    pub fn new(device: &Device, cmd: &CommandPool, data: &[V]) -> Result<GpuFuture<Self>, vk::Result> {
        let mut slf = Self::with_capacity(device, data.len())?;
        slf.cpu.map_copy_unmap(0, data)?;
        let fence = Fence::new(device, false)?;
        cmd.create_command_buffer()?
            .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?
            .copy(&slf.cpu, &slf.gpu)
            .end()?
            .submit(&[], &[], Some(&fence))?;
        Ok(GpuFuture::new(slf, fence))
    }
}

pub type VertexBuffer<V: VertexSource> = StageBuffer<V, Cpu, Gpu>;

impl<V: Debug> VertexBuffer<V> {
    pub fn new_vertex_buffer(device: &Device, cmd: &CommandPool, data: &[V]) -> Result<GpuFuture<Self>, vk::Result> {
        Self::new(device, cmd, data)
    }
}

pub type IndirectBuffer = StageBuffer<vk::DrawIndirectCommand, Cpu, GpuIndirect>;

impl IndirectBuffer {
    pub fn new_indirect_buffer(device: &Device, cmd: &CommandPool, data: &[vk::DrawIndirectCommand]) -> Result<GpuFuture<Self>, vk::Result> {
        Self::new(device, cmd, data)
    }
}