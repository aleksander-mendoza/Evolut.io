use ash::vk;
use crate::surface::Surface;
use crate::window::{WINDOW_WIDTH, WINDOW_HEIGHT};
use crate::device::Device;
use ash::version::DeviceV1_0;
use crate::imageview::ImageView;

pub struct SwapChain {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    device: Device
}

fn choose_swapchain_format(available_formats: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
    for available_format in available_formats {
        if available_format.format == vk::Format::B8G8R8A8_SRGB
            && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return available_format.clone();
        }
    }
    return available_formats.first().unwrap().clone();
}

fn choose_swapchain_present_mode(available_present_modes: &Vec<vk::PresentModeKHR>) -> vk::PresentModeKHR {
    for &available_present_mode in available_present_modes.iter() {
        if available_present_mode == vk::PresentModeKHR::MAILBOX {
            return available_present_mode;
        }
    }

    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: WINDOW_WIDTH.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: WINDOW_HEIGHT.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

impl SwapChain {
    pub fn new(instance: &ash::Instance, device: &Device, surface: &Surface) -> Result<Self, failure::Error> {
        let capabilities = surface.capabilities(device.physical_device())?;
        let formats = surface.formats(device.physical_device())?;
        let present_modes = surface.present_modes(device.physical_device())?;


        let surface_format = choose_swapchain_format(&formats);
        let present_mode = choose_swapchain_present_mode(&present_modes);
        let extent = choose_swapchain_extent(&capabilities);

        let image_count = capabilities.min_image_count + 1;
        let image_count = if capabilities.max_image_count > 0 {
            image_count.min(capabilities.max_image_count)
        } else {
            image_count
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.raw())
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null())
            .image_array_layers(1)
            .present_mode(present_mode);

        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device.device());
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }?;

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }?;

        Ok(Self {
            swapchain_loader,
            swapchain,
            format: surface_format.format,
            extent,
            images,
            device:device.clone()
        })
    }

    pub fn create_image_views(&self) -> Result<Vec<ImageView>,vk::Result> {
        self.images.iter().map(|&image|ImageView::new(image,self.format,self.device())).collect()
    }

    pub fn format(&self)-> vk::Format{
        self.format
    }
    pub fn extent(&self)-> vk::Extent2D{
        self.extent
    }
    pub fn device(&self)-> &Device{
        &self.device
    }
}

impl Drop for SwapChain{
    fn drop(&mut self) {
        unsafe{
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}