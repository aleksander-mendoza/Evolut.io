use crate::render::buffer_type::{BufferType, AsDescriptor};
use std::rc::Rc;
use crate::render::owned_buffer::OwnedBuffer;
use crate::render::device::Device;
use ash::vk;
use std::collections::Bound;
use std::ops::RangeBounds;
use crate::render::buffer::Buffer;
use ash::vk::DeviceMemory;

pub struct SubBuffer<V: Copy, T: BufferType>{
    buff:Rc<OwnedBuffer<V, T>>,
    offset:usize,
    size:usize
}

impl <V: Copy, T: BufferType> SubBuffer<V,T>{
    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self, ash::vk::Result> {
        OwnedBuffer::with_capacity(device,capacity).map(Self::from)
    }
}

impl <V: Copy, T: BufferType> From<OwnedBuffer<V,T>> for SubBuffer<V,T>{
    fn from(b: OwnedBuffer<V, T>) -> Self {
        let size = b.capacity();
        Self{buff:Rc::new(b), offset: 0, size }
    }
}

impl <V: Copy, T: BufferType> Buffer<V,T> for SubBuffer<V,T> {
    fn device(&self) -> &Device {
        self.parent().device()
    }
    fn raw(&self) -> vk::Buffer {
        self.parent().raw()
    }
    fn capacity(&self) -> usize {
        self.len()
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn len(&self) -> usize {
        self.size
    }

    fn raw_mem(&self) -> DeviceMemory {
        self.parent().raw_mem()
    }
}

impl <V: Copy, T: BufferType>  SubBuffer<V,T>{
    fn parent(&self) -> &OwnedBuffer<V, T> {
        &self.buff
    }
    pub fn sub(&self, range:impl RangeBounds<usize>) -> Self{
        let from = match range.start_bound(){
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i+1,
            Bound::Unbounded => 0
        };
        let to = match range.start_bound(){
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i-1,
            Bound::Unbounded => self.size
        };
        assert!(from<=to);
        assert!(to<=self.len());
        Self{buff:self.buff.clone(), offset:self.offset+from, size:to-from}
    }


}

