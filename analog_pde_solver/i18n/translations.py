"""Multi-language support for the analog PDE solver system."""

import os
import json
from typing import Dict, Optional, Any
from pathlib import Path
from ..utils.logging_config import get_logger


class TranslationManager:
    """Manages multi-language translations for the analog PDE solver."""
    
    def __init__(self, default_language: str = "en"):
        """Initialize translation manager.
        
        Args:
            default_language: Default language code (ISO 639-1)
        """
        self.default_language = default_language
        self.current_language = default_language
        self.logger = get_logger('i18n')
        
        # Translation cache
        self.translations: Dict[str, Dict[str, str]] = {}
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'es': 'Español', 
            'fr': 'Français',
            'de': 'Deutsch',
            'ja': '日本語',
            'zh': '中文',
            'ko': '한국어',
            'pt': 'Português',
            'it': 'Italiano',
            'ru': 'Русский'
        }
        
        # Load default translations
        self._load_default_translations()
        
        self.logger.info(f"Translation manager initialized (default: {default_language})")
    
    def set_language(self, language_code: str) -> bool:
        """Set the current language.
        
        Args:
            language_code: ISO 639-1 language code
            
        Returns:
            True if language was set successfully
        """
        if language_code in self.supported_languages:
            self.current_language = language_code
            self.logger.info(f"Language changed to: {language_code}")
            return True
        else:
            self.logger.warning(f"Unsupported language: {language_code}")
            return False
    
    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages."""
        return self.supported_languages.copy()
    
    def translate(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """Translate a message key to the specified or current language.
        
        Args:
            key: Translation key
            language: Target language (uses current if None)
            **kwargs: Format arguments for the translation
            
        Returns:
            Translated string
        """
        target_lang = language or self.current_language
        
        # Get translation
        translation = self._get_translation(key, target_lang)
        
        # Format with provided arguments
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Translation formatting error for key '{key}': {e}")
        
        return translation
    
    def _get_translation(self, key: str, language: str) -> str:
        """Get translation for key in specified language."""
        # Check if language is loaded
        if language not in self.translations:
            self._load_language(language)
        
        # Get translation from target language
        if language in self.translations and key in self.translations[language]:
            return self.translations[language][key]
        
        # Fallback to default language
        if (self.default_language in self.translations and 
            key in self.translations[self.default_language]):
            return self.translations[self.default_language][key]
        
        # Final fallback - return key itself
        self.logger.warning(f"Missing translation: {key} ({language})")
        return key
    
    def _load_language(self, language: str):
        """Load translations for a specific language."""
        if language in self.translations:
            return  # Already loaded
        
        # Try to load from file (would be implemented in production)
        translations_file = Path(__file__).parent / f"locales/{language}.json"
        
        if translations_file.exists():
            try:
                with open(translations_file, 'r', encoding='utf-8') as f:
                    self.translations[language] = json.load(f)
                self.logger.info(f"Loaded translations for: {language}")
            except Exception as e:
                self.logger.error(f"Failed to load translations for {language}: {e}")
                self.translations[language] = {}
        else:
            # Use default translations for missing languages
            self.translations[language] = self._get_default_translations(language)
    
    def _load_default_translations(self):
        """Load default translations for all supported languages."""
        for lang_code in self.supported_languages:
            self.translations[lang_code] = self._get_default_translations(lang_code)
    
    def _get_default_translations(self, language: str) -> Dict[str, str]:
        """Get default translations for a language."""
        translations_db = {
            'en': {
                # System messages
                'system.startup': 'Analog PDE Solver System Starting',
                'system.shutdown': 'System Shutting Down',
                'system.ready': 'System Ready',
                'system.error': 'System Error: {error}',
                
                # Solver messages
                'solver.initializing': 'Initializing analog PDE solver...',
                'solver.programming_crossbar': 'Programming crossbar arrays...',
                'solver.solving': 'Solving PDE with {iterations} iterations',
                'solver.converged': 'Solution converged after {iterations} iterations',
                'solver.failed_convergence': 'Failed to converge after {iterations} iterations',
                'solver.error': 'Solver error: {error}',
                
                # Health monitoring
                'health.status_healthy': 'System health: HEALTHY',
                'health.status_warning': 'System health: WARNING',
                'health.status_critical': 'System health: CRITICAL',
                'health.crossbar_health': 'Crossbar health: {health}%',
                'health.solver_performance': 'Solver performance: {performance}%',
                'health.memory_usage': 'Memory usage: {usage} MB',
                
                # Performance monitoring
                'perf.optimization_enabled': 'Performance optimization enabled',
                'perf.caching_hit': 'Cache hit: {hit_rate}% hit rate',
                'perf.scaling_up': 'Scaling up: adding {instances} instances',
                'perf.scaling_down': 'Scaling down: removing {instances} instances',
                
                # Error messages
                'error.invalid_parameters': 'Invalid parameters provided',
                'error.convergence_failed': 'Convergence failed',
                'error.memory_insufficient': 'Insufficient memory',
                'error.hardware_failure': 'Hardware failure detected',
                'error.configuration_invalid': 'Invalid configuration',
                
                # Status messages
                'status.initializing': 'Initializing...',
                'status.ready': 'Ready',
                'status.running': 'Running',
                'status.stopping': 'Stopping...',
                'status.stopped': 'Stopped',
                'status.error': 'Error',
                
                # Configuration
                'config.loading': 'Loading configuration...',
                'config.loaded': 'Configuration loaded successfully',
                'config.validation_failed': 'Configuration validation failed',
                'config.applying': 'Applying configuration changes...',
                
                # Hardware
                'hw.crossbar_programming': 'Programming crossbar array ({size}x{size})',
                'hw.analog_computation': 'Performing analog computation',
                'hw.digital_interface': 'Interfacing with digital systems',
                'hw.calibration': 'Performing hardware calibration',
                
                # Validation messages
                'validation.domain_size_invalid': 'Invalid domain size: must be positive integer',
                'validation.conductance_range_invalid': 'Invalid conductance range',
                'validation.noise_model_unsupported': 'Unsupported noise model: {model}',
                'validation.crossbar_size_invalid': 'Invalid crossbar size: {size}',
                
                # Units and measurements
                'units.seconds': 'seconds',
                'units.milliseconds': 'milliseconds',
                'units.microseconds': 'microseconds',
                'units.megabytes': 'MB',
                'units.gigabytes': 'GB',
                'units.milliwatts': 'mW',
                'units.nanjoules': 'nJ',
                'units.siemens': 'S',
                'units.iterations': 'iterations',
                'units.percent': '%'
            },
            
            'es': {
                # System messages
                'system.startup': 'Sistema Solucionador PDE Analógico Iniciando',
                'system.shutdown': 'Sistema Cerrando',
                'system.ready': 'Sistema Listo',
                'system.error': 'Error del Sistema: {error}',
                
                # Solver messages
                'solver.initializing': 'Inicializando solucionador PDE analógico...',
                'solver.programming_crossbar': 'Programando arreglos crossbar...',
                'solver.solving': 'Resolviendo PDE con {iterations} iteraciones',
                'solver.converged': 'Solución convergió después de {iterations} iteraciones',
                'solver.failed_convergence': 'Falló convergencia después de {iterations} iteraciones',
                'solver.error': 'Error del solucionador: {error}',
                
                # Health monitoring
                'health.status_healthy': 'Estado del sistema: SALUDABLE',
                'health.status_warning': 'Estado del sistema: ADVERTENCIA',
                'health.status_critical': 'Estado del sistema: CRÍTICO',
                'health.crossbar_health': 'Estado crossbar: {health}%',
                'health.solver_performance': 'Rendimiento solucionador: {performance}%',
                'health.memory_usage': 'Uso de memoria: {usage} MB',
                
                # Error messages
                'error.invalid_parameters': 'Parámetros inválidos proporcionados',
                'error.convergence_failed': 'Convergencia falló',
                'error.memory_insufficient': 'Memoria insuficiente',
                'error.hardware_failure': 'Falla de hardware detectada',
                'error.configuration_invalid': 'Configuración inválida',
                
                # Status messages
                'status.initializing': 'Inicializando...',
                'status.ready': 'Listo',
                'status.running': 'Ejecutando',
                'status.stopping': 'Deteniendo...',
                'status.stopped': 'Detenido',
                'status.error': 'Error',
                
                # Units
                'units.seconds': 'segundos',
                'units.milliseconds': 'milisegundos',
                'units.megabytes': 'MB',
                'units.iterations': 'iteraciones',
                'units.percent': '%'
            },
            
            'fr': {
                # System messages
                'system.startup': 'Système Solveur EDP Analogique Démarrage',
                'system.shutdown': 'Arrêt du Système',
                'system.ready': 'Système Prêt',
                'system.error': 'Erreur Système: {error}',
                
                # Solver messages
                'solver.initializing': 'Initialisation du solveur EDP analogique...',
                'solver.programming_crossbar': 'Programmation des réseaux crossbar...',
                'solver.solving': 'Résolution EDP avec {iterations} itérations',
                'solver.converged': 'Solution convergée après {iterations} itérations',
                'solver.failed_convergence': 'Échec de convergence après {iterations} itérations',
                'solver.error': 'Erreur du solveur: {error}',
                
                # Health monitoring
                'health.status_healthy': 'État du système: SAIN',
                'health.status_warning': 'État du système: AVERTISSEMENT',
                'health.status_critical': 'État du système: CRITIQUE',
                'health.crossbar_health': 'Santé crossbar: {health}%',
                'health.solver_performance': 'Performance solveur: {performance}%',
                'health.memory_usage': 'Utilisation mémoire: {usage} MB',
                
                # Status messages
                'status.initializing': 'Initialisation...',
                'status.ready': 'Prêt',
                'status.running': 'En cours',
                'status.stopping': 'Arrêt...',
                'status.stopped': 'Arrêté',
                'status.error': 'Erreur',
                
                # Units
                'units.seconds': 'secondes',
                'units.milliseconds': 'millisecondes',
                'units.megabytes': 'MB',
                'units.iterations': 'itérations',
                'units.percent': '%'
            },
            
            'de': {
                # System messages
                'system.startup': 'Analoger PDE-Löser System Startet',
                'system.shutdown': 'System Herunterfahren',
                'system.ready': 'System Bereit',
                'system.error': 'Systemfehler: {error}',
                
                # Solver messages
                'solver.initializing': 'Analoger PDE-Löser wird initialisiert...',
                'solver.programming_crossbar': 'Crossbar-Arrays werden programmiert...',
                'solver.solving': 'PDE-Lösung mit {iterations} Iterationen',
                'solver.converged': 'Lösung konvergiert nach {iterations} Iterationen',
                'solver.failed_convergence': 'Konvergenz fehlgeschlagen nach {iterations} Iterationen',
                'solver.error': 'Löserfehler: {error}',
                
                # Health monitoring
                'health.status_healthy': 'Systemzustand: GESUND',
                'health.status_warning': 'Systemzustand: WARNUNG',
                'health.status_critical': 'Systemzustand: KRITISCH',
                'health.crossbar_health': 'Crossbar-Zustand: {health}%',
                'health.solver_performance': 'Löser-Leistung: {performance}%',
                'health.memory_usage': 'Speicherverbrauch: {usage} MB',
                
                # Status messages
                'status.initializing': 'Initialisierung...',
                'status.ready': 'Bereit',
                'status.running': 'Läuft',
                'status.stopping': 'Stoppe...',
                'status.stopped': 'Gestoppt',
                'status.error': 'Fehler',
                
                # Units
                'units.seconds': 'Sekunden',
                'units.milliseconds': 'Millisekunden',
                'units.megabytes': 'MB',
                'units.iterations': 'Iterationen',
                'units.percent': '%'
            },
            
            'ja': {
                # System messages
                'system.startup': 'アナログPDE解法システム開始中',
                'system.shutdown': 'システム終了中',
                'system.ready': 'システム準備完了',
                'system.error': 'システムエラー: {error}',
                
                # Solver messages
                'solver.initializing': 'アナログPDE解法器を初期化中...',
                'solver.programming_crossbar': 'クロスバー配列をプログラミング中...',
                'solver.solving': '{iterations}回の反復でPDEを解法中',
                'solver.converged': '{iterations}回の反復後に解が収束',
                'solver.failed_convergence': '{iterations}回の反復後に収束失敗',
                'solver.error': '解法器エラー: {error}',
                
                # Health monitoring
                'health.status_healthy': 'システム状態: 正常',
                'health.status_warning': 'システム状態: 警告',
                'health.status_critical': 'システム状態: 危険',
                'health.crossbar_health': 'クロスバー状態: {health}%',
                'health.solver_performance': '解法器性能: {performance}%',
                'health.memory_usage': 'メモリ使用量: {usage} MB',
                
                # Status messages
                'status.initializing': '初期化中...',
                'status.ready': '準備完了',
                'status.running': '実行中',
                'status.stopping': '停止中...',
                'status.stopped': '停止',
                'status.error': 'エラー',
                
                # Units
                'units.seconds': '秒',
                'units.milliseconds': 'ミリ秒',
                'units.megabytes': 'MB',
                'units.iterations': '回',
                'units.percent': '%'
            },
            
            'zh': {
                # System messages
                'system.startup': '模拟偏微分方程求解系统启动中',
                'system.shutdown': '系统关闭中',
                'system.ready': '系统就绪',
                'system.error': '系统错误: {error}',
                
                # Solver messages
                'solver.initializing': '正在初始化模拟偏微分方程求解器...',
                'solver.programming_crossbar': '正在编程交叉开关阵列...',
                'solver.solving': '使用{iterations}次迭代求解偏微分方程',
                'solver.converged': '解在{iterations}次迭代后收敛',
                'solver.failed_convergence': '{iterations}次迭代后收敛失败',
                'solver.error': '求解器错误: {error}',
                
                # Health monitoring
                'health.status_healthy': '系统状态: 健康',
                'health.status_warning': '系统状态: 警告',
                'health.status_critical': '系统状态: 严重',
                'health.crossbar_health': '交叉开关健康度: {health}%',
                'health.solver_performance': '求解器性能: {performance}%',
                'health.memory_usage': '内存使用: {usage} MB',
                
                # Status messages
                'status.initializing': '初始化中...',
                'status.ready': '就绪',
                'status.running': '运行中',
                'status.stopping': '停止中...',
                'status.stopped': '已停止',
                'status.error': '错误',
                
                # Units
                'units.seconds': '秒',
                'units.milliseconds': '毫秒',
                'units.megabytes': 'MB',
                'units.iterations': '次迭代',
                'units.percent': '%'
            }
        }
        
        # Return translations for the requested language, fallback to English
        return translations_db.get(language, translations_db['en'])
    
    def add_custom_translation(self, language: str, key: str, value: str):
        """Add a custom translation.
        
        Args:
            language: Language code
            key: Translation key
            value: Translation value
        """
        if language not in self.translations:
            self.translations[language] = {}
        
        self.translations[language][key] = value
        self.logger.debug(f"Added custom translation: {language}.{key}")
    
    def get_language_name(self, language_code: str) -> str:
        """Get the display name for a language code."""
        return self.supported_languages.get(language_code, language_code)
    
    def format_number(self, number: float, language: Optional[str] = None) -> str:
        """Format number according to locale conventions.
        
        Args:
            number: Number to format
            language: Target language (uses current if None)
            
        Returns:
            Formatted number string
        """
        target_lang = language or self.current_language
        
        # Simple number formatting by language
        if target_lang in ['en', 'ja', 'zh', 'ko']:
            # Use period for decimal, comma for thousands
            return f"{number:,.2f}"
        elif target_lang in ['de', 'fr', 'es', 'it', 'pt', 'ru']:
            # Use comma for decimal, space/period for thousands
            formatted = f"{number:,.2f}"
            return formatted.replace(',', ' ').replace('.', ',')
        else:
            return f"{number:.2f}"
    
    def format_duration(self, seconds: float, language: Optional[str] = None) -> str:
        """Format duration with appropriate units and language.
        
        Args:
            seconds: Duration in seconds
            language: Target language
            
        Returns:
            Formatted duration string
        """
        if seconds < 0.001:
            return f"{seconds*1000000:.1f} {self.translate('units.microseconds', language)}"
        elif seconds < 1.0:
            return f"{seconds*1000:.1f} {self.translate('units.milliseconds', language)}"
        else:
            return f"{seconds:.2f} {self.translate('units.seconds', language)}"
    
    def format_memory(self, bytes_val: int, language: Optional[str] = None) -> str:
        """Format memory size with appropriate units.
        
        Args:
            bytes_val: Memory size in bytes
            language: Target language
            
        Returns:
            Formatted memory string
        """
        if bytes_val >= 1024**3:
            return f"{bytes_val/(1024**3):.2f} {self.translate('units.gigabytes', language)}"
        else:
            return f"{bytes_val/(1024**2):.1f} {self.translate('units.megabytes', language)}"


# Global translation manager instance
_translation_manager = None

def get_translation_manager() -> TranslationManager:
    """Get the global translation manager instance."""
    global _translation_manager
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    return _translation_manager

def t(key: str, language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translation.
    
    Args:
        key: Translation key
        language: Target language
        **kwargs: Format arguments
        
    Returns:
        Translated string
    """
    return get_translation_manager().translate(key, language, **kwargs)

def set_language(language_code: str) -> bool:
    """Set the global language.
    
    Args:
        language_code: ISO 639-1 language code
        
    Returns:
        True if successful
    """
    return get_translation_manager().set_language(language_code)