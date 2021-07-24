use crate::render::frames_in_flight::FramesInFlight;
use crate::render::device::Device;
use ash::vk::{PhysicalDevice};
use crate::render::surface::Surface;
use crate::render::instance::Instance;
use winit::event_loop::EventLoop;
use crate::render::window::init_window;
use winit::window::Window;

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
    window: Window,
    entry: ash::Entry,
}

impl VulkanContext {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self, failure::Error> {
        let entry = unsafe { ash::Entry::new() }?;
        let window = init_window(event_loop)?;
        let instance = Instance::new(&entry, true)?;
        let surface = instance.create_surface(&entry, &window)?;
        let physical_device = instance.pick_physical_device(&surface)?;
        let device = instance.create_device(&entry, physical_device)?;
        let mut frames_in_flight = FramesInFlight::new(&device, 2)?;
        Ok(Self {
            entry,
            window,
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
    pub fn window(&self) -> &Window {
        &self.window
    }
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }
}