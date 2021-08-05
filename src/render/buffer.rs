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

pub trait DeviceLocal: Type {}

pub trait AsDescriptor: Type {}

pub trait AsStorage: AsDescriptor {}

impl AsDescriptor for Uniform{}

pub struct Uniform {}

impl Type for Uniform {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw());
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
}


impl CpuWriteable for Uniform {}

pub struct Gpu {}

impl DeviceLocal for Gpu{}

impl Type for Gpu {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::VERTEX_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}

impl DeviceLocal for Storage{}

impl AsDescriptor for Storage{}

impl AsStorage for Storage{}

pub struct Storage {}

impl Type for Storage {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::STORAGE_BUFFER.as_raw() | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}


/**It's just like STORAGE buffer, but it does not have TRANSFER_DST flag, because it's meant to be initialised and used only on device*/
pub struct ProceduralStorage {}

impl Type for ProceduralStorage {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::STORAGE_BUFFER.as_raw());
}
impl AsDescriptor for ProceduralStorage{}
impl AsStorage for ProceduralStorage{}
impl GpuWriteable for Storage {}
impl GpuWriteable for Gpu {}

pub struct GpuIndirect {}

impl Type for GpuIndirect {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw() | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()  | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}

impl GpuWriteable for GpuIndirect {}
impl DeviceLocal for GpuIndirect {}
impl AsDescriptor for GpuIndirect{}
impl AsStorage for GpuIndirect {}

pub struct Cpu {}

impl Type for Cpu {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw());
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;
}

impl CpuWriteable for Cpu {}


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
    pub fn make_shader_buffer_barrier(&self)->vk::BufferMemoryBarrier{
        self.make_buffer_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ)
    }
    pub fn make_buffer_barrier(&self, src_access_mask: vk::AccessFlags, dst_access_mask: vk::AccessFlags)->vk::BufferMemoryBarrier{
        vk::BufferMemoryBarrier::builder()
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .buffer(self.raw())
            .offset(0)
            .size(self.len() as u64)
            .build()
    }
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


impl<V: Copy, T: AsDescriptor> Buffer<V, T> {
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
    pub unsafe fn map(&mut self, offset: usize, len: usize) -> Result<*mut V, vk::Result> {
        assert!(offset + len <= self.capacity);
        self.device.inner().map_memory(
            self.mem,
            offset as u64,
            len as u64,
            vk::MemoryMapFlags::empty(),
        ).map(|v| v as *mut V)
    }
    pub unsafe fn unmap(&mut self) {
        self.device.inner().unmap_memory(self.mem)
    }
    unsafe fn unsafe_map_unmap(&mut self, offset: usize, len: usize, f: impl FnOnce(*mut V)) -> Result<(), vk::Result> {
        f(self.map(offset, len)?);
        self.unmap();
        Ok(())
    }
}
