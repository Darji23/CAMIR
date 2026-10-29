"""Unit tests for per-profile energy tracking plugins."""

import time
import pytest
from camir.accounting.energy import (
    AppleSiliconEnergyPlugin,
    CloudEnergyPlugin,
    EnergyTracker,
    NvidiaEnergyPlugin,
    RaplEnergyPlugin,
    get_energy_plugin,
)


def test_energy_plugin_factory():
    apple_plugin = get_energy_plugin("apple_silicon")
    assert isinstance(apple_plugin, AppleSiliconEnergyPlugin)
    assert apple_plugin.profile_name == "apple_silicon"

    nv_plugin = get_energy_plugin("nvidia")
    assert isinstance(nv_plugin, NvidiaEnergyPlugin)
    assert nv_plugin.profile_name == "nvidia"

    rapl_plugin = get_energy_plugin("cpu_rapl")
    assert isinstance(rapl_plugin, RaplEnergyPlugin)
    assert rapl_plugin.profile_name == "cpu_rapl"

    cloud_plugin = get_energy_plugin("cloud")
    assert isinstance(cloud_plugin, CloudEnergyPlugin)
    assert cloud_plugin.profile_name == "cloud"


def test_apple_silicon_estimated_energy():
    plugin = AppleSiliconEnergyPlugin(use_powermetrics=False)
    plugin.start()
    time.sleep(0.05)
    energy_j, method, label = plugin.stop()

    assert energy_j is not None
    assert energy_j > 0.0
    assert method == "estimated"
    assert "12.5W" in label


def test_cloud_energy_plugin_returns_none():
    plugin = CloudEnergyPlugin()
    plugin.start()
    energy_j, method, label = plugin.stop()

    assert energy_j is None
    assert method == "none"
    assert "cloud" in label.lower()


def test_energy_tracker_context_manager():
    with EnergyTracker(profile="apple_silicon") as tracker:
        time.sleep(0.02)

    assert tracker.energy_j is not None
    assert tracker.energy_j > 0.0
    assert tracker.energy_method == "estimated"
    assert "12.5W" in tracker.label
