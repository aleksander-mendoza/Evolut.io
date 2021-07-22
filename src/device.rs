use ash::version::{InstanceV1_0, DeviceV1_0};
use ash::vk;
use failure::err_msg;
use ash::vk::{QueueFamilyProperties, ExtensionProperties, PhysicalDeviceMemoryProperties};
use crate::instance::Instance;
use crate::validation_layer::get_validation_layer_support;
use crate::surface::Surface;
use std::ffi::CStr;
use std::rc::Rc;
use ash::prelude::VkResult;

#[cfg(not(target_os = "macos"))]
fn device_extensions() -> [*const i8; 1] {
    [ash::extensions::khr::Swapchain::name().as_ptr()]
}
#[cfg(target_os = "macos")]
fn device_extensions() -> [*const i8; 2] {
    [ash::extensions::khr::Swapchain::name().as_ptr(),
        b"VK_KHR_portability_subset\0".as_ptr() as *const i8]
}


pub fn extension_name(ext: &ExtensionProperties) -> &CStr {
    unsafe {
        CStr::from_ptr(ext.extension_name.as_ptr())
    }
}

fn has_necessary_queues(queue_family: &QueueFamilyProperties) -> bool {
    queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
}

fn score_physical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> u32 {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    //let device_features = unsafe { instance.get_physical_device_features(physical_device) };
    let device_queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let available_extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) };
    let available_extensions = match available_extensions {
        Ok(available_extensions) => available_extensions,
        Err(_) => return 0
    };
    let available_extensions: Vec<&CStr> = available_extensions.iter().map(extension_name).collect();
    let device_type_score = match device_properties.device_type {
        vk::PhysicalDeviceType::CPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::OTHER => 0,
        _ => 0
    };
    if device_type_score == 0 { return 0; }
    if surface.formats(physical_device).map(|v| v.len()).unwrap_or(0) == 0 { return 0; }
    if surface.present_modes(physical_device).map(|v| v.len()).unwrap_or(0) == 0 { return 0; }
    if !device_extensions().iter().all(|&extension| available_extensions.contains(&unsafe { CStr::from_ptr(extension) })) {
        return 0;
    }
    let queue_score = device_queue_families.iter().enumerate()
        .map(|(idx, fam)| if has_necessary_queues(fam) && surface.supported_by(physical_device, idx as u32).unwrap_or(false) { 1 } else { 0 })
        .max().unwrap_or(0);

    device_type_score * queue_score
}

pub fn pick_physical_device(instance: &ash::Instance, surface: &Surface) -> Result<ash::vk::PhysicalDevice, failure::Error> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }?;

    println!("Devices found with vulkan support:\n{:?}", physical_devices);

    physical_devices
        .iter()
        .map(|&dev| (dev, score_physical_device(instance, dev, surface)))
        .max_by_key(|(dev, score)| *score)
        .and_then(|(dev, score)| if score > 0 { Some(dev) } else { None })
        .ok_or_else(|| err_msg("No suitable devices are available. You need to have a GPU with compute shaders and graphics pipeline"))
}

fn pick_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> u32 {
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    queue_families
        .iter()
        .position(has_necessary_queues)
        .expect("This should never happen if the physical device was picked in the first place") as u32
}

struct DeviceInner {
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    instance: Instance,
    family_index: u32,
}

#[derive(Clone)]
pub struct Device {
    inner: Rc<DeviceInner>,
}

impl Device {
    pub fn new(entry: &ash::Entry, instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<Self, failure::Error> {
        let family_index = pick_queue_family(instance.raw(), physical_device);

        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(family_index)
            .queue_priorities(&[1.0]);

        let features = vk::PhysicalDeviceFeatures::builder();

        let layers = get_validation_layer_support(entry)?;
        let extensions = device_extensions();
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_layer_names(layers)
            .enabled_features(&features)
            .enabled_extension_names(&extensions);

        let device = unsafe { instance.raw().create_device(physical_device, &device_create_info, None) }?;

        let queue = unsafe { device.get_device_queue(family_index, 0) };

        Ok(Self { inner: Rc::new(DeviceInner { device, instance:instance.clone(), queue, family_index, physical_device }) })
    }
    pub fn family_index(&self) -> u32 {
        self.inner.family_index
    }
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.physical_device
    }
    pub fn raw(&self) -> ash::vk::Device {
        self.inner.device.handle()
    }
    pub fn raw_queue(&self) -> vk::Queue{
        self.inner.queue
    }
    pub fn inner(&self) -> &ash::Device {
        &self.inner.device
    }
    pub fn instance(&self) -> &Instance {
        &self.inner.instance
    }
    pub fn device_wait_idle(&self) -> VkResult<()> {
        unsafe { self.inner().device_wait_idle() }
    }
    pub fn queue_wait_idle(&self) -> VkResult<()> {
        unsafe { self.inner().queue_wait_idle(self.inner.queue) }
    }
    pub fn get_physical_device_memory_properties(&self) -> PhysicalDeviceMemoryProperties {
        unsafe { self.instance().raw().get_physical_device_memory_properties(self.physical_device()) }
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None)
        }
    }
}