"""
PyTorch-native caching system with automatic invalidation based on parameter versions.
"""

from threading import RLock
from typing import Callable, List, Optional, Tuple, Any
import torch.nn as nn


class ParameterVersionTracker:
    """
    Tracks versions of PyTorch parameters to detect changes.

    """
    
    def __init__(self):
        self._tracked_params: List[Tuple[str, nn.Parameter]] = []
        self._last_state: Optional[Tuple[Tuple[int, int, float], ...]] = None
    
    def track(self, name: str, param: nn.Parameter):
        """Register a parameter to track."""
        self._tracked_params.append((name, param))
        self._last_state = None
    
    def track_module(self, module: nn.Module):
        """Track all parameters in a module."""
        for name, param in module.named_parameters():
            self.track(name, param)
    
    def _get_param_state(self, param: nn.Parameter) -> Tuple[int, int, float]:
        """Get a state tuple for a parameter that changes when the param changes."""
        version = param._version
        data_ptr = param.data.data_ptr()
        checksum = param.data.sum().item()
        return (version, data_ptr, checksum)
    
    def get_state(self) -> Tuple[Tuple[int, int, float], ...]:
        """Get current state of all tracked parameters."""
        return tuple(self._get_param_state(p) for _, p in self._tracked_params)
    
    def has_changed(self) -> bool:
        """Check if any tracked parameter has changed since last check."""
        current = self.get_state()
        return self._last_state != current
    
    def mark_clean(self):
        """Mark current state as clean (cache is valid)."""
        self._last_state = self.get_state()
    
    def invalidate(self):
        """Force invalidation on next check."""
        self._last_state = None


class versioned_cached_property:
    """
    A cached property that automatically invalidates when tracked parameters change.
    
    This is a DATA DESCRIPTOR (has __set__), which means __get__ is always called
    on attribute access, allowing us to check for parameter changes every time.
    
    The cached values are stored in instance._versioned_property_cache to avoid
    conflicts with the descriptor protocol.
    
    Usage:
        class MyModule(nn.Module, VersionedCacheMixin):
            def __init__(self):
                super().__init__()
                VersionedCacheMixin.__init__(self)
                self.param = nn.Parameter(torch.randn(3, 3))
                self._version_tracker.track("param", self.param)
            
            @versioned_cached_property
            def expensive_computation(self):
                return self.param @ self.param.T
    """
    
    def __init__(self, func: Callable):
        self.func = func
        self.attrname: Optional[str] = None
        self.__doc__ = func.__doc__
        self.lock = RLock()
    
    def __set_name__(self, owner, name: str):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                f"Cannot assign the same versioned_cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )
    
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        
        if self.attrname is None:
            raise TypeError(
                "Cannot use versioned_cached_property instance without calling __set_name__ on it."
            )
        
        # Get or create the property cache
        try:
            prop_cache = instance._versioned_property_cache
        except AttributeError:
            prop_cache = {}
            instance._versioned_property_cache = prop_cache
        
        # Check if parameters changed
        tracker = getattr(instance, '_version_tracker', None)
        if tracker is not None and tracker.has_changed():
            prop_cache.clear()
            # Also clear dict-based cache if present
            if hasattr(instance, '_cache') and isinstance(instance._cache, dict):
                instance._cache.clear()
            tracker.mark_clean()
        
        # Check cache
        if self.attrname not in prop_cache:
            with self.lock:
                if self.attrname not in prop_cache:
                    prop_cache[self.attrname] = self.func(instance)
        
        return prop_cache[self.attrname]
    
    def __set__(self, instance, value):
        """
        Make this a data descriptor so __get__ is always called.
        
        Allows explicit setting of the cached value if needed.
        """
        try:
            prop_cache = instance._versioned_property_cache
        except AttributeError:
            prop_cache = {}
            instance._versioned_property_cache = prop_cache
        prop_cache[self.attrname] = value
    
    def __delete__(self, instance):
        """Allow deletion of cached value."""
        try:
            del instance._versioned_property_cache[self.attrname]
        except (AttributeError, KeyError):
            pass


class VersionedCacheMixin:
    
    def __init__(self):
        self._version_tracker = ParameterVersionTracker()
        self._cache: dict = {}
        self._versioned_property_cache: dict = {}
    
    def _check_cache_validity(self) -> bool:
        """Check if cache is still valid. If not, clear it."""
        if self._version_tracker.has_changed():
            self._invalidate_caches()
            self._version_tracker.mark_clean()
            return False
        return True
    
    def _invalidate_caches(self):
        """Clear all caches."""
        self._cache.clear()
        self._versioned_property_cache.clear()
    
    def _get_cached(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """
        Get a cached value, computing it if necessary.
        
        Automatically invalidates if parameters have changed.
        """
        self._check_cache_validity()
        
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]
    
    def clear_cache(self):
        """Manually clear all caches and force recomputation."""
        self._version_tracker.invalidate()
        self._invalidate_caches()


class HierarchicalVersionTracker(ParameterVersionTracker):
    """
    Version tracker that can track child modules' parameters.
    """
    
    def __init__(self):
        super().__init__()
        self._child_trackers: List[ParameterVersionTracker] = []
    
    def track_child(self, child_tracker: ParameterVersionTracker):
        """Track a child's version tracker."""
        self._child_trackers.append(child_tracker)
        self._last_state = None
    
    def get_state(self) -> Tuple:
        """Get state from self and all children."""
        own_state = super().get_state()
        child_states = tuple(
            s for tracker in self._child_trackers 
            for s in tracker.get_state()
        )
        return own_state + child_states


def track_parameters_from_module(tracker: ParameterVersionTracker, module: nn.Module):
    """Helper to track all parameters from a module."""
    for name, param in module.named_parameters(recurse=True):
        tracker.track(name, param)


def create_child_aware_tracker(*children: 'VersionedCacheMixin') -> HierarchicalVersionTracker:
    """Create a tracker that monitors child modules."""
    tracker = HierarchicalVersionTracker()
    for child in children:
        if hasattr(child, '_version_tracker'):
            tracker.track_child(child._version_tracker)
    return tracker