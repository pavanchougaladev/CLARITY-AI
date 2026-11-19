"""
Registration History Manager
============================
Stores metrics and results of past registration runs in a JSON database
to allow tracking improvements over time and comparing parameter settings.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, storage_file: str = "registration_history.json", max_entries: int = 500):
        self.storage_file = storage_file
        self.max_entries = max_entries
        self._history = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        """Load history from JSON file."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading history: {e}")
        return []

    def _save(self):
        """Save history to JSON file, keeping only the max_entries most recent."""
        # Keep only the most recent N entries
        if len(self._history) > self.max_entries:
            self._history = self._history[-self.max_entries:]
            
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving history: {e}")

    def add_entry(self, 
                  src_path: str, 
                  ref_path: str, 
                  method: str, 
                  metrics: Dict[str, Any], 
                  params: Dict[str, Any] = None):
        """
        Add a new registration run to the history.
        """
        entry = {
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'src_basename': os.path.basename(src_path),
            'ref_basename': os.path.basename(ref_path),
            'method': method,
            'metrics': {
                'rmse_pixels': float(metrics['rmse_pixels']) if metrics.get('rmse_pixels') is not None else None,
                'weighted_rmse': float(metrics['weighted_rmse']) if metrics.get('weighted_rmse') is not None else None,
                'inlier_ratio': float(metrics.get('inlier_ratio', 0)),
                'inlier_count': int(metrics.get('inlier_count', 0)),
                'total_matches': int(metrics.get('total_matches', 0)),
                'coverage': float(metrics.get('coverage', 0)),
                'grade': metrics.get('quality', {}).get('grade', 'N/A') if isinstance(metrics.get('quality'), dict) else metrics.get('grade', 'N/A'),
                'passed': metrics.get('quality', {}).get('passed', True) if isinstance(metrics.get('quality'), dict) else metrics.get('passed', True),
            },
            'params': params or {}
        }
        
        self._history.append(entry)
        self._save()

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieve the full history (most recent last)."""
        return self._history

    def get_best_runs(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top N runs sorted by RMSE."""
        valid_runs = [h for h in self._history if h['metrics']['rmse_pixels'] is not None]
        sorted_runs = sorted(valid_runs, key=lambda x: x['metrics']['rmse_pixels'])
        return sorted_runs[:top_n]
    
    def clear(self):
        """Clear the history completely."""
        self._history = []
        self._save()
