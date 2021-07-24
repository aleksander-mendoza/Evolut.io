use crate::render::platforms::create_surface;
use ash::vk;
use ash::prelude::VkResult;
use ash::vk::{SurfaceCapabilitiesKHR, SurfaceFormatKHR, PresentModeKHR};

pub struct Surface {
    surface_loader: ash::extensions::khr::Surface,
    raw: vk::SurfaceKHR,
}

impl Surface {
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.raw
    }
    pub fn new(entry: &ash::Entry,
               instance: &ash::Instance,
               window: &winit::window::Window) -> Result<Self, failure::Error> {
        let surface = unsafe { create_surface(entry, instance, window) }?;
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

        Ok(Self {
            surface_loader,
            raw: surface,
        })
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