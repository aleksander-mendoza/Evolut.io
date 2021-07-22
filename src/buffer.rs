use ash::vk;
use crate::data::VertexSource;
use crate::device::Device;
use ash::version::{DeviceV1_0, InstanceV1_0};
use crate::instance::Instance;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use crate::command_pool::CommandPool;
use crate::fence::Fence;
use crate::gpu_future::GpuFuture;



pub trait Type {
    const SHARING_MODE: vk::SharingMode;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags;
    const USAGE: vk::BufferUsageFlags;
}

pub trait CpuWriteable:Type{}

pub struct Uniform {}

impl Type for Uniform {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw());
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
}

impl CpuWriteable for Uniform{}

pub struct Gpu {}

impl Type for Gpu {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(vk::BufferUsageFlags::VERTEX_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw());
}


pub struct Cpu {}

impl Type for Cpu {
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
    const REQUIRED_MEMORY_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw());
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;
}

impl CpuWriteable for Cpu{}

pub struct Buffer<V: VertexSource, T:Type> {
    capacity: usize,
    raw: vk::Buffer,
    mem: vk::DeviceMemory,
    device: Device,
    _u: PhantomData<T>,
    _v: PhantomData<V>,
}

impl<V: VertexSource, T:Type> Drop for Buffer<V, T> {
    fn drop(&mut self) {
        unsafe {
            self.device.inner().destroy_buffer(self.raw, None);
            self.device.inner().free_memory(self.mem, None);
        }
    }
}

impl<V: VertexSource, T: Type> Buffer<V, T> {
    pub fn device(&self)->&Device{
        &self.device
    }
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn mem_capacity(&self) -> vk::DeviceSize{
        (std::mem::size_of::<V>() * self.capacity) as u64
    }
    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self, vk::Result> {
        let vertex_buffer_create_info = vk::BufferCreateInfo::builder()
            .size((std::mem::size_of::<V>() * capacity) as u64)
            .usage(T::USAGE)
            .sharing_mode(T::SHARING_MODE);

        let vertex_buffer = unsafe { device.inner().create_buffer(&vertex_buffer_create_info, None) }?;

        let mem_requirements = unsafe { device.inner().get_buffer_memory_requirements(vertex_buffer) };
        let mem_properties = device.get_physical_device_memory_properties();
        let memory_type = Self::find_memory_type(
            mem_requirements.memory_type_bits,
            T::REQUIRED_MEMORY_FLAGS,
            mem_properties,
        );

        let allocate_info = vk::MemoryAllocateInfo::builder()
            .memory_type_index(memory_type)
            .allocation_size(mem_requirements.size);

        let vertex_buffer_memory = unsafe { device.inner().allocate_memory(&allocate_info, None) }?;
        unsafe{
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

    fn find_memory_type(
        type_filter: u32,
        required_properties: vk::MemoryPropertyFlags,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
            // same implementation
            if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(required_properties) {
                return i as u32;
            }
        }

        panic!("Failed to find suitable memory type!")
    }
}

impl<V: VertexSource> Buffer<V, Uniform> {
    pub fn descriptor_info(&self)->vk::DescriptorBufferInfo{
        vk::DescriptorBufferInfo {
            buffer: self.raw(),
            offset: 0,
            range: self.mem_capacity(),
        }
    }
}
impl<V: VertexSource, T:CpuWriteable> Buffer<V, T> {
    pub fn new(device: &Device, data:&[V]) -> Result<Self, vk::Result> {
        let mut slf = Self::with_capacity(device, data.len())?;
        slf.map_copy_unmap(0,data);
        Ok(slf)
    }

    pub fn map_copy_unmap(&mut self, offset: usize, data: &[V]) -> Result<(), vk::Result> {
        assert!(offset+data.len()<=self.capacity);
        unsafe {
            let data_ptr = self.device.inner().map_memory(
                self.mem,
                offset as u64,
                data.len() as u64,
                vk::MemoryMapFlags::empty(),
            )? as *mut V;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

            self.device.inner().unmap_memory(self.mem);
            Ok(())
        }
    }


}


pub struct PairedBuffers<V: VertexSource>{
    cpu:Buffer<V,Cpu>,
    gpu:Buffer<V,Gpu>
}

impl <V: VertexSource> PairedBuffers<V>{
    pub fn cpu(&self)->&Buffer<V,Cpu>{
        &self.cpu
    }
    pub fn gpu(&self)->&Buffer<V,Gpu>{
        &self.gpu
    }
    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self,vk::Result>{
        let cpu = Buffer::with_capacity(device,capacity)?;
        let gpu = Buffer::with_capacity(device,capacity)?;
        Ok(Self{cpu,gpu})
    }

    pub fn new(device: &Device, cmd:&CommandPool, fence:Fence, data:&[V]) -> Result<GpuFuture<Self>,vk::Result>{
        let mut slf = Self::with_capacity(device,data.len())?;
        slf.cpu.map_copy_unmap(0,data)?;
        cmd.create_command_buffer()?
            .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?
            .copy(&slf.cpu, &slf.gpu)
            .end()?
            .submit(&[],&[], Some(&fence))?;
        Ok(GpuFuture::new(slf,fence))
    }
}