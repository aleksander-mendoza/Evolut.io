use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use crate::render_pass::RenderPass;
use crate::swap_chain::SwapChain;
use std::marker::PhantomData;
use crate::framebuffer::Framebuffer;
use ash::vk::ImageUsageFlags;


pub trait Aspect{
    const ASPECT:vk::ImageAspectFlags;
    const USAGE:vk::ImageUsageFlags;
}

pub struct Color{}
pub struct Depth{}
impl Aspect for Color{
    const ASPECT:vk::ImageAspectFlags=vk::ImageAspectFlags::COLOR;
    const USAGE: ImageUsageFlags = vk::ImageUsageFlags::from_raw(vk::ImageUsageFlags::TRANSFER_DST.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw());
}
impl Aspect for Depth{
    const ASPECT:vk::ImageAspectFlags=vk::ImageAspectFlags::DEPTH;
    const USAGE: ImageUsageFlags = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
}

pub struct ImageView<A:Aspect> {
    raw: vk::ImageView,
    device: Device,
    _a:PhantomData<A>
}

impl <A:Aspect> ImageView<A> {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn raw(&self) -> vk::ImageView {
        self.raw
    }
    pub fn new(raw: vk::Image, format: vk::Format, device: &Device) -> Result<Self, ash::vk::Result> {
        let imageview_create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: A::ASPECT,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image(raw);
        unsafe { device.inner().create_image_view(&imageview_create_info, None) }.map(|img| Self { raw: img, device: device.clone(), _a: PhantomData })
    }
}
impl ImageView<Color>{
    pub fn create_framebuffer(&self, render_pass: &RenderPass, swapchain: &SwapChain) -> Result<Framebuffer, ash::vk::Result> {
        Framebuffer::new(render_pass,swapchain,&[self.raw()])
    }

}

impl <A:Aspect> Drop for ImageView<A> {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_image_view(self.raw, None); }
    }
}