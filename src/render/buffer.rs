use crate::render::buffer_type::{BufferType, AsDescriptor, CpuWriteable, AsStorage};
use crate::render::device::Device;
use ash::vk;
use ash::version::DeviceV1_0;

pub trait Buffer<V: Copy, T: BufferType> {
    fn device(&self) -> &Device;
    fn raw(&self) -> vk::Buffer;
    fn capacity(&self) -> usize;
    fn offset(&self) -> usize;
    fn len(&self) -> usize;
    fn mem_capacity(&self) -> vk::DeviceSize {
        (std::mem::size_of::<V>() * self.capacity()) as u64
    }
    fn raw_mem(&self) -> vk::DeviceMemory;
}

pub fn descriptor_info<V: Copy, T: AsDescriptor>(buff: &impl Buffer<V, T>) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo {
        buffer: buff.raw(),
        offset: buff.offset() as u64,
        range: buff.mem_capacity(),
    }
}


pub fn map_unmap_whole<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, f: impl FnOnce(&mut [V])) -> Result<(), vk::Result> {
    map_unmap(buff, buff.offset(), buff.len(), f)
}

pub fn map_copy_unmap<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, offset: usize, data: &[V]) -> Result<(), vk::Result> {
    unsafe {
        unsafe_map_unmap(buff, offset, data.len(), |ptr| ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()))
    }
}

pub fn map_unmap<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, offset: usize, len: usize, f: impl FnOnce(&mut [V])) -> Result<(), vk::Result> {
    unsafe {
        unsafe_map_unmap(buff, offset, len, |ptr| f(std::slice::from_raw_parts_mut(ptr, len)))
    }
}

pub unsafe fn map_whole<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>) -> Result<*mut V, vk::Result> {
    map(buff, buff.offset(), buff.capacity())
}

pub unsafe fn map<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, offset: usize, len: usize) -> Result<*mut V, vk::Result> {
    assert!(offset + len <= buff.capacity());
    buff.device().inner().map_memory(
        buff.raw_mem(),
        (buff.offset() + offset) as u64,
        len as u64,
        vk::MemoryMapFlags::empty(),
    ).map(|v| v as *mut V)
}

pub unsafe fn unmap<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>) {
    buff.device().inner().unmap_memory(buff.raw_mem())
}

unsafe fn unsafe_map_unmap<V: Copy, T: CpuWriteable>(buff: &mut impl Buffer<V, T>, offset: usize, len: usize, f: impl FnOnce(*mut V)) -> Result<(), vk::Result> {
    f(map(buff,offset, len)?);
    unmap(buff);
    Ok(())
}

pub fn make_shader_buffer_barrier<V: Copy, T: AsStorage>(buff: &impl Buffer<V, T>) -> vk::BufferMemoryBarrier {
    make_buffer_barrier(buff, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ)
}

pub fn make_buffer_barrier<V: Copy, T: AsStorage>(buff: &impl Buffer<V, T>, src_access_mask: vk::AccessFlags, dst_access_mask: vk::AccessFlags) -> vk::BufferMemoryBarrier {
    vk::BufferMemoryBarrier::builder()
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .buffer(buff.raw())
        .offset(buff.offset() as u64)
        .size(buff.len() as u64)
        .build()
}
