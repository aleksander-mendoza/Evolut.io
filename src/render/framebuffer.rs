use ash::vk;
use crate::render::device::Device;
use crate::render::render_pass::RenderPass;
use crate::render::swap_chain::SwapChain;
use crate::render::imageview::{ImageView, Depth, Color};

use failure::err_msg;

pub struct Framebuffer {
    raw: vk::Framebuffer,
    device: Device,
    render_pass: RenderPass,
    image_view:ImageView<Color>
}

impl Framebuffer {
    pub fn raw(&self) -> vk::Framebuffer {
        self.raw
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn new(render_pass: &RenderPass, swapchain: &SwapChain, color:ImageView<Color>, depth:&ImageView<Depth>) -> Result<Framebuffer, failure::Error> {
        let attachments = [color.raw(),depth.raw()];
        let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.raw())
            .attachments(&attachments)
            .width(swapchain.extent().width)
            .height(swapchain.extent().height)
            .layers(1);

        unsafe {
            render_pass.device().inner().create_framebuffer(&framebuffer_create_info, None)
        }.map(move |raw| Framebuffer { raw, device: render_pass.device().clone(), image_view: color, render_pass: render_pass.clone() }).map_err(err_msg)
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_framebuffer(self.raw, None); }
    }
}