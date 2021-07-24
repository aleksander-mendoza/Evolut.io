use ash::vk;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use ash::prelude::VkResult;

pub struct Fence {
    raw: vk::Fence,
    device: Device,
}

impl Fence {
    pub fn new(device: &Device, signaled: bool) -> Result<Self, ash::vk::Result> {
        let flags = if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() };
        let fence_create_info = vk::FenceCreateInfo::builder().flags(flags);
        unsafe { device.inner().create_fence(&fence_create_info, None) }.map(|raw| Self { raw, device: device.clone() })
    }

    pub fn wait(&self, timeout:Option<u64>) -> VkResult<()> {
        unsafe{self.device.inner().wait_for_fences(&[self.raw], true, timeout.unwrap_or(u64::MAX))}
    }

    pub fn reset(&self) -> VkResult<()> {
        unsafe{self.device.inner().reset_fences(&[self.raw])}
    }

    pub fn raw(&self)->vk::Fence{
        self.raw
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_fence(self.raw, None); }
    }
}