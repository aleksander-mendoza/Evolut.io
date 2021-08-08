use crate::render::frames_in_flight::FramesInFlight;
use crate::render::device::Device;
use ash::vk::{PhysicalDevice};
use crate::render::surface::Surface;
use crate::render::instance::Instance;
use crate::render::single_render_pass::SingleRenderPass;

pub struct VulkanContext {
    // The order of all fields
    // is very important, because
    // they will be dropped
    // in the exact same order
    frames_in_flight: FramesInFlight,
    device: Device,
    physical_device: PhysicalDevice,
    surface: Surface,
    instance: Instance,
    entry: ash::Entry,
}

impl VulkanContext {
    pub fn new(window: sdl2::video::Window) -> Result<Self, failure::Error> {
        let entry = unsafe { ash::Entry::new() }?;
        let instance = Instance::new(&entry, true)?;
        let surface = instance.create_surface(&entry, window)?;
        let physical_device = instance.pick_physical_device(&surface)?;
        let device = instance.create_device(&entry, physical_device)?;
        let frames_in_flight = FramesInFlight::new(&device, 2)?;
        Ok(Self {
            entry,
            instance,
            surface,
            physical_device,
            device,
            frames_in_flight,
        })
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn frames_in_flight(&self) -> &FramesInFlight {
        &self.frames_in_flight
    }
    pub fn frames_in_flight_mut(&mut self) -> &mut FramesInFlight {
        &mut self.frames_in_flight
    }
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }
    pub fn surface(&self) -> &Surface {
        &self.surface
    }
    pub fn instance(&self) -> &Instance {
        &self.instance
    }
    pub fn window(&self) -> &sdl2::video::Window {
        &self.surface.window()
    }
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }
    pub fn create_single_render_pass(&self)->Result<SingleRenderPass,failure::Error>{
        SingleRenderPass::new_swapchain_and_render_pass(self.instance(),self.device(),self.surface())
    }
}