use crate::render::fence::Fence;
use ash::vk;
use crate::render::command_pool::{CommandPool, CommandBuffer};
use std::ops::{Deref, DerefMut};
use ash::prelude::VkResult;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use crate::render::owned_buffer::{OwnedBuffer};
use crate::render::buffer_type::BufferType;
use crate::render::buffer::Buffer;

pub struct SubmitterCmd {
    pool: CommandPool,
    cmd: CommandBuffer,
    fence: Fence,
    was_submitted: bool,
}

impl SubmitterCmd {
    pub fn device(&self) -> &Device {
        &self.pool.device()
    }
    pub fn wait(&self, timeout: Option<u64>) -> VkResult<()> {
        if self.was_submitted {
            self.fence.wait(timeout)
        } else {
            Ok(())
        }
    }
    pub fn cmd_pool(&self) -> &CommandPool {
        &self.pool
    }
    pub fn cmd(&mut self) -> &mut CommandBuffer {
        &mut self.cmd
    }
    pub fn reset(&mut self) -> VkResult<&mut CommandBuffer> {
        self.wait(None);
        self.fence.reset()?;
        self.was_submitted = false;
        Ok(self.cmd())
    }
    pub fn submit(&mut self) -> VkResult<()> {
        self.cmd.submit(&[], &[], Some(&self.fence))?;
        self.was_submitted = true;
        Ok(())
    }
    pub fn new(pool: &CommandPool) -> Result<Self, vk::Result> {
        Ok(Self { fence: Fence::new(pool.device(), false)?, pool: pool.clone(), cmd: pool.create_command_buffer()?, was_submitted: false })
    }
}

pub struct Submitter<T> {
    inner: SubmitterCmd,
    val: T,
}

impl Drop for SubmitterCmd {
    fn drop(&mut self) {
        self.wait(None).expect("Failed to wait during Submitter destruction");
        unsafe {
            self.device().inner().free_command_buffers(self.pool.raw(), &[self.cmd.raw()])
        }
    }
}

impl<T> Deref for Submitter<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl<T> DerefMut for Submitter<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.val
    }
}

impl<T> Submitter<T> {
    pub fn new(val: T, pool: &CommandPool) -> Result<Self, vk::Result> {
        SubmitterCmd::new(pool).map(move |inner| Self { inner, val })
    }
    pub fn take(self) -> VkResult<T> {
        self.inner().wait(None)?;
        let Self { inner, val } = self;
        Ok(val)
    }
    pub fn inner_val(&mut self) -> (&mut SubmitterCmd, &mut T) {
        let Self { inner, val } = self;
        (inner, val)
    }
    pub fn inner(&self) -> &SubmitterCmd {
        &self.inner
    }
    pub fn wait(&self, timeout: Option<u64>) -> VkResult<()> {
        self.inner().wait(timeout)
    }
    pub fn reset(&mut self) -> VkResult<&mut CommandBuffer> {
        self.inner_mut().reset()
    }
    pub fn inner_mut(&mut self) -> &mut SubmitterCmd {
        &mut self.inner
    }
    pub fn cmd_pool(&self) -> &CommandPool {
        self.inner().cmd_pool()
    }
    pub fn cmd(&mut self) -> &mut CommandBuffer {
        self.inner_mut().cmd()
    }
    pub fn device(&self) -> &Device {
        self.inner().device()
    }
}

pub fn fill_submit<T: BufferType, B: Buffer<u32, T>>(sub: &mut Submitter<B>, val: u32) -> VkResult<()> {
    let (inner, buff) = sub.inner_val();
    inner.cmd().begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?.fill(buff, val).end()?;
    inner.submit()
}

pub fn fill_zeros_submit<V: Copy, T: BufferType, B: Buffer<V, T>>(sub: &mut Submitter<B>) -> VkResult<()> {
    let (inner, val) = sub.inner_val();
    inner.cmd().begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?.fill_zeros(val).end()?;
    inner.submit()
}