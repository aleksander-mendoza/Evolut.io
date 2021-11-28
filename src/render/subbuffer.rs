use crate::render::buffer_type::{BufferType};

use crate::render::owned_buffer::OwnedBuffer;
use crate::render::device::Device;
use ash::vk;
use std::collections::Bound;
use std::ops::RangeBounds;
use crate::render::buffer::Buffer;
use ash::vk::DeviceMemory;
use std::sync::Arc;

pub struct SubBuffer<V: Copy, T: BufferType>{
    buff:Arc<OwnedBuffer<V, T>>,
    offset:vk::DeviceSize,
    size:vk::DeviceSize
}
impl <V: Copy, T: BufferType> Clone for SubBuffer<V,T>{
    fn clone(&self) -> Self {
        Self{buff:self.buff.clone(), offset:self.offset, size:self.size}
    }
}

impl <V: Copy, T: BufferType> SubBuffer<V,T>{

    pub fn reinterpret_into<V2:Copy>(self) -> SubBuffer<V2, T> {
        unsafe{std::mem::transmute::<SubBuffer<V,T>,SubBuffer<V2,T>>(self)}
    }
    pub fn reinterpret_as<V2:Copy>(&self) -> &SubBuffer<V2, T> {
        unsafe{std::mem::transmute::<&SubBuffer<V,T>,&SubBuffer<V2,T>>(self)}
    }
    pub fn parent(&self) -> &OwnedBuffer<V, T> {
        &self.buff
    }
    pub fn element(&self, idx:vk::DeviceSize) -> Self{
        let s = self.element_size() as u64;
        self.sub(idx*s..(idx+1)*s)
    }
    pub fn sub_elem(&self, offset:vk::DeviceSize,len:vk::DeviceSize) -> Self{
        let o = offset*std::mem::size_of::<V>() as u64;
        self.sub(o..o+len*std::mem::size_of::<V>() as u64)
    }
    pub fn sub(&self, range:impl RangeBounds<vk::DeviceSize>) -> Self{
        let from = match range.start_bound(){
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i+1,
            Bound::Unbounded => 0
        };
        let to = match range.end_bound(){
            Bound::Included(&i) => i+1,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.size
        };
        assert!(from<=to);
        assert!(to<=self.bytes());
        Self{buff:self.buff.clone(), offset:self.offset+from, size:to-from}
    }

}


impl <V: Copy, T: BufferType> From<OwnedBuffer<V,T>> for SubBuffer<V,T>{
    fn from(b: OwnedBuffer<V, T>) -> Self {
        let size = b.bytes();
        Self{buff:Arc::new(b), offset: 0, size }
    }
}

impl <V: Copy, T: BufferType> Buffer<V,T> for SubBuffer<V,T> {
    fn device(&self) -> &Device {
        self.parent().device()
    }
    fn raw(&self) -> vk::Buffer {
        self.parent().raw()
    }

    fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    fn bytes(&self) -> vk::DeviceSize {
        self.size
    }

    fn raw_mem(&self) -> DeviceMemory {
        self.parent().raw_mem()
    }

    fn with_capacity(device: &Device, max_elements: vk::DeviceSize) -> Result<Self, ash::vk::Result> {
        OwnedBuffer::with_capacity(device,max_elements).map(Self::from)
    }
}
