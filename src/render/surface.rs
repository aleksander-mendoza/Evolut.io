use ash::vk;
use ash::prelude::VkResult;
use ash::vk::{SurfaceCapabilitiesKHR, SurfaceFormatKHR, PresentModeKHR, SurfaceKHR, Handle};
use crate::render::instance::Instance;
use failure::err_msg;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::dpi::PhysicalSize;

pub struct Surface {
    surface_loader: ash::extensions::khr::Surface,
    raw: vk::SurfaceKHR,
    window: winit::window::Window
}

impl Surface {
    pub fn window(&self)->&winit::window::Window{
        &self.window
    }
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.raw
    }
    pub fn new(entry: &ash::Entry,
               instance: &Instance,
               window: winit::window::Window) -> Result<Self, failure::Error> {
        let surface =  unsafe { ash_window::create_surface(entry, instance.raw(), window.raw_display_handle(), window.raw_window_handle(),None).unwrap() };
        // window.vulkan_create_surface( instance.raw().handle().as_raw() as usize).map_err(err_msg)?;
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance.raw());
        let surface = SurfaceKHR::from_raw(surface.as_raw());
        Ok(Self {
            surface_loader,
            raw: surface,
            window,
        })
    }
    pub fn size(&self) -> (u32, u32) {
        let PhysicalSize{ width, height } = self.window.inner_size();
        (width, height)
    }
    pub fn supported_by(&self, device: vk::PhysicalDevice, family_idx: u32) -> VkResult<bool> {
        unsafe {
            self.surface_loader.get_physical_device_surface_support(device, family_idx, self.raw)
        }
    }
    pub fn capabilities(&self, physical_device: ash::vk::PhysicalDevice) -> VkResult<SurfaceCapabilitiesKHR> {
        unsafe {
            self.surface_loader.get_physical_device_surface_capabilities(physical_device, self.raw)
        }
    }
    pub fn formats(&self, physical_device: ash::vk::PhysicalDevice) -> VkResult<Vec<SurfaceFormatKHR>> {
        unsafe {
            self.surface_loader.get_physical_device_surface_formats(physical_device, self.raw)
        }
    }
    pub fn present_modes(&self, physical_device: ash::vk::PhysicalDevice) -> VkResult<Vec<PresentModeKHR>> {
        unsafe {
            self.surface_loader.get_physical_device_surface_present_modes(physical_device, self.raw)
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.raw, None);
        }
    }
}