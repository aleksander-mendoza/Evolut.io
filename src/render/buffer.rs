use crate::render::buffer_type::{BufferType, AsDescriptor, CpuWriteable, AsStorage, GpuIndirect};
use crate::render::device::Device;
use ash::vk;

pub trait Buffer<V: Copy, T: BufferType>:Sized {
    fn device(&self) -> &Device;
    fn raw(&self) -> vk::Buffer;
    fn offset(&self) -> vk::DeviceSize;
    fn len(&self) -> vk::DeviceSize{
        self.bytes() / self.element_size() as u64
    }
    fn bytes(&self) -> vk::DeviceSize;
    fn element_offset(&self, idx:u64) -> vk::DeviceSize {
        assert!(idx<self.len(),"idx={}, len={}, offset={}",idx,self.len(),self.offset());
        self.offset()+self.element_size() as u64 * idx
    }
    fn raw_mem(&self) -> vk::DeviceMemory;
    fn with_capacity(device: &Device, max_elements: vk::DeviceSize) -> Result<Self, vk::Result>;
    fn element_size(&self)->usize{
        std::mem::size_of::<V>()
    }
}

pub fn descriptor_info<V: Copy, T: AsDescriptor>(buff: &impl Buffer<V, T>) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo {
        buffer: buff.raw(),
        offset: buff.offset(),
        range: buff.bytes(),
    }
}


pub fn map_unmap_whole<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, f: impl FnOnce(&mut [V])) -> Result<(), vk::Result> {
    map_unmap(buff, buff.offset(), buff.bytes(), f)
}

pub fn map_copy_unmap<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, offset: vk::DeviceSize, data: &[V]) -> Result<(), vk::Result> {
    unsafe {
        unsafe_map_unmap(buff, offset, data.len() as u64, |ptr| ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()))
    }
}

pub fn map_unmap<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, offset: vk::DeviceSize, len: vk::DeviceSize, f: impl FnOnce(&mut [V])) -> Result<(), vk::Result> {
    unsafe {
        unsafe_map_unmap(buff, offset, len, |ptr| f(std::slice::from_raw_parts_mut(ptr, len as usize)))
    }
}

pub unsafe fn map_whole<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>) -> Result<*mut V, vk::Result> {
    map(buff, buff.offset(), buff.bytes())
}

pub unsafe fn map<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, offset: vk::DeviceSize, len: vk::DeviceSize) -> Result<*mut V, vk::Result> {
    assert!(offset + len <= buff.bytes());
    assert_eq!(len % buff.element_size() as u64,0,"Len: {} Type: {}",len, std::any::type_name::<V>() );
    buff.device().inner().map_memory(
        buff.raw_mem(),
        buff.offset() + offset,
        len,
        vk::MemoryMapFlags::empty(),
    ).map(|v| v as *mut V)
}

pub unsafe fn unmap<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>) {
    buff.device().inner().unmap_memory(buff.raw_mem())
}

unsafe fn unsafe_map_unmap<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, offset: vk::DeviceSize, len: vk::DeviceSize, f: impl FnOnce(*mut V)) -> Result<(), vk::Result> {
    f(map(buff,offset, len)?);
    unmap(buff);
    Ok(())
}

pub fn make_shader_buffer_barrier<V: Copy, T: AsStorage>(buff: &impl Buffer<V, T>) -> vk::BufferMemoryBarrier {
    make_buffer_barrier(buff, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ)
}

pub fn make_shader_dispatch_buffer_barrier<V: Copy>(buff: &impl Buffer<V, GpuIndirect>) -> vk::BufferMemoryBarrier {
    make_buffer_barrier(buff, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::INDIRECT_COMMAND_READ)
}

pub fn make_buffer_barrier<V: Copy, T: AsStorage>(buff: &impl Buffer<V, T>, src_access_mask: vk::AccessFlags, dst_access_mask: vk::AccessFlags) -> vk::BufferMemoryBarrier {
    vk::BufferMemoryBarrier::builder()
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .buffer(buff.raw())
        .offset(buff.offset() as u64)
        .size(buff.bytes() as u64)
        .build()
}
