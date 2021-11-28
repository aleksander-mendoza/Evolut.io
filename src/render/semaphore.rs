use ash::vk;
use crate::render::device::Device;
use ash::version::DeviceV1_0;

pub struct Semaphore{
    raw:vk::Semaphore,
    device:Device
}

impl Semaphore{
    pub fn new(device:&Device) -> Result<Self, ash::vk::Result> {
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
        unsafe{device.inner().create_semaphore(&semaphore_create_info, None)}.map(|raw|Self{raw,device:device.clone()})
    }
    pub fn raw(&self)->vk::Semaphore{
        self.raw
    }
}

impl Drop for Semaphore{
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_semaphore(self.raw, None); }
    }
}