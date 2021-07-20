use ash::vk;
use crate::data::VertexSource;
use crate::device::Device;
use ash::version::{DeviceV1_0, InstanceV1_0};
use crate::instance::Instance;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;

pub trait Usage{
    const USAGE:vk::BufferUsageFlags;
}

pub struct VertexBuffer{
}

impl Usage for VertexBuffer{
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::VERTEX_BUFFER;
}

pub trait SharingMode{
    const SHARING_MODE:vk::SharingMode;
}


pub struct Exclusive{
}

impl SharingMode for Exclusive{
    const SHARING_MODE: vk::SharingMode = vk::SharingMode::EXCLUSIVE;
}

pub struct Buffer<V:VertexSource, U:Usage, S:SharingMode>{
    data:Vec<V>,
    raw:vk::Buffer,
    mem:vk::DeviceMemory,
    device:Device,
    _u:PhantomData<U>,
    _s:PhantomData<S>
}

impl <V:VertexSource, U:Usage, S:SharingMode> Drop for Buffer<V, U, S> {
    fn drop(&mut self) {
        unsafe {
            self.device.inner().destroy_buffer(self.raw, None);
            self.device.inner().free_memory(self.mem, None);
        }
    }
}

impl <V:VertexSource, U:Usage, S:SharingMode> Deref for Buffer<V, U, S> {
    type Target = Vec<V>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl <V:VertexSource, U:Usage, S:SharingMode> DerefMut for Buffer<V, U, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl <V:VertexSource, U:Usage, S:SharingMode> Buffer<V, U, S>{

    pub fn raw(&self)->vk::Buffer{
        self.raw
    }
    pub fn new(device: &Device, data:impl Into<Vec<V>>) -> Result<Self, vk::Result>{
        let mut slf = Self::from_vec(device,data.into())?;
        slf.flush_all();
        Ok(slf)
    }
    pub fn with_capacity(device: &Device, capacity:usize) -> Result<Self, vk::Result>{
        Self::from_vec(device,Vec::with_capacity(capacity))
    }
    fn from_vec(device: &Device, vec:Vec<V>) -> Result<Self, vk::Result>{
        let vertex_buffer_create_info = vk::BufferCreateInfo::builder()
            .size((std::mem::size_of::<V>() * vec.capacity()) as u64)
            .usage(U::USAGE)
            .sharing_mode(S::SHARING_MODE);

        let vertex_buffer = unsafe { device.inner().create_buffer(&vertex_buffer_create_info, None) }?;

        let mem_requirements = unsafe { device.inner().get_buffer_memory_requirements(vertex_buffer) };
        let mem_properties = device.get_physical_device_memory_properties();
        let required_memory_flags: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = Self::find_memory_type(
            mem_requirements.memory_type_bits,
            required_memory_flags,
            mem_properties,
        );

        let allocate_info = vk::MemoryAllocateInfo::builder()
            .memory_type_index(memory_type)
            .allocation_size(mem_requirements.size);

        let vertex_buffer_memory = unsafe { device.inner().allocate_memory(&allocate_info, None) }?;

        let slf = Self{
            data: vec,
            raw: vertex_buffer,
            mem: vertex_buffer_memory,
            device: device.clone(),
            _u: PhantomData,
            _s: PhantomData
        };
        Ok(slf)
    }
    pub fn flush_all(&mut self)->Result<(),vk::Result>{
        self.flush(0,self.data.len())
    }
    pub fn flush(&mut self, offset:usize, len:usize)->Result<(),vk::Result>{
        unsafe {
            self.device.inner().bind_buffer_memory(self.raw, self.mem, 0)?;

            let data_ptr = self.device.inner().map_memory(
                    self.mem,
                    offset as u64,
                    len as u64,
                    vk::MemoryMapFlags::empty(),
                )? as *mut V;

            data_ptr.copy_from_nonoverlapping(self.data.as_ptr(), self.data.len());

            self.device.inner().unmap_memory(self.mem);
            Ok(())
        }
    }

    fn find_memory_type(
        type_filter: u32,
        required_properties: vk::MemoryPropertyFlags,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
            //if (type_filter & (1 << i)) > 0 && (memory_type.property_flags & required_properties) == required_properties {
            //    return i as u32
            // }

            // same implementation
            if (type_filter & (1 << i)) > 0
                && memory_type.property_flags.contains(required_properties)
            {
                return i as u32;
            }
        }

        panic!("Failed to find suitable memory type!")
    }
}
