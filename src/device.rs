use ash::version::{InstanceV1_0, DeviceV1_0};
use ash::vk;
use failure::err_msg;
use ash::vk::QueueFamilyProperties;
use crate::instance::Instance;
use crate::validation_layer::get_validation_layer_support;

fn score_queue_family(queue_family: &QueueFamilyProperties) -> u32 {
    if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) { 1u32 } else { 0 }
}

fn score_physical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> u32 {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    //let device_features = unsafe { instance.get_physical_device_features(physical_device) };
    let device_queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let device_type_score = match device_properties.device_type {
        vk::PhysicalDeviceType::CPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::OTHER => 0,
        _ => 0
    };
    let queue_score = device_queue_families.iter().map(score_queue_family).max().unwrap_or(0);

    device_type_score * queue_score
}

pub fn pick_physical_device(instance: &ash::Instance) -> Result<ash::vk::PhysicalDevice, failure::Error> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }?;

    println!("Devices found with vulkan support:\n{:?}", physical_devices);

    physical_devices
        .iter()
        .map(|&dev| (dev, score_physical_device(instance, dev)))
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
        .position(|fam| score_queue_family(fam) == 1)
        .expect("This should never happen if the physical device was picked in the first place") as u32
}

pub struct Device {
    device: ash::Device,
    queue: vk::Queue,
}

impl Device {
    pub fn new(entry:&ash::Entry, instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Result<Self, failure::Error> {
        let family_idx = pick_queue_family(instance, physical_device);

        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(family_idx)
            .queue_priorities(&[1.0]);

        let features = vk::PhysicalDeviceFeatures::builder();

        let layers = get_validation_layer_support(entry)?;

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_layer_names(layers)
            .enabled_features(&features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }?;

        let queue = unsafe { device.get_device_queue(family_idx, 0) };

        Ok(Self{device, queue})
    }
}

impl Drop for Device{
    fn drop(&mut self) {
         unsafe{
             self.device.destroy_device(None)
         }
    }
}