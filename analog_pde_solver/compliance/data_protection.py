"""Data protection and privacy compliance for analog PDE solver."""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from ..utils.logging_config import get_logger


class DataClassification(Enum):
    """Data classification levels for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ProcessingPurpose(Enum):
    """Legal purposes for data processing."""
    COMPUTATION = "computation"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    DEBUGGING = "debugging"
    ANALYTICS = "analytics"
    RESEARCH = "research"


@dataclass
class DataSubject:
    """Information about a data subject (user/entity)."""
    subject_id: str
    subject_type: str  # 'user', 'organization', 'system'
    jurisdiction: str  # ISO 3166-1 alpha-2 country code
    consent_date: Optional[float] = None
    consent_purposes: Set[ProcessingPurpose] = field(default_factory=set)
    retention_period_days: int = 365
    
    def has_valid_consent(self, purpose: ProcessingPurpose) -> bool:
        """Check if subject has valid consent for purpose."""
        return purpose in self.consent_purposes


@dataclass
class DataRecord:
    """Record of data processing activity."""
    record_id: str
    subject_id: str
    data_type: str
    classification: DataClassification
    purpose: ProcessingPurpose
    processing_date: float
    retention_until: float
    jurisdiction: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if data retention period has expired."""
        return time.time() > self.retention_until
    
    def anonymize(self) -> 'DataRecord':
        """Create anonymized version of record."""
        anon_record = DataRecord(
            record_id=self.record_id,
            subject_id="anonymized",
            data_type=self.data_type,
            classification=DataClassification.INTERNAL,
            purpose=self.purpose,
            processing_date=self.processing_date,
            retention_until=self.retention_until,
            jurisdiction=self.jurisdiction,
            metadata={'anonymized': True}
        )
        return anon_record


class DataProtectionManager:
    """Manages data protection compliance (GDPR, CCPA, PDPA, etc.)."""
    
    def __init__(self, organization_name: str, default_jurisdiction: str = "EU"):
        """Initialize data protection manager.
        
        Args:
            organization_name: Name of processing organization
            default_jurisdiction: Default jurisdiction for compliance
        """
        self.organization_name = organization_name
        self.default_jurisdiction = default_jurisdiction
        self.logger = get_logger('data_protection')
        
        # Data subject registry
        self.data_subjects: Dict[str, DataSubject] = {}
        
        # Processing records
        self.processing_records: List[DataRecord] = []
        
        # Compliance rules by jurisdiction
        self.compliance_rules = self._load_compliance_rules()
        
        self.logger.info(f"Data protection manager initialized for {organization_name}")
    
    def register_data_subject(
        self,
        subject_id: str,
        subject_type: str = "user",
        jurisdiction: str = None
    ) -> DataSubject:
        """Register a new data subject.
        
        Args:
            subject_id: Unique identifier for subject
            subject_type: Type of subject
            jurisdiction: Subject's jurisdiction
            
        Returns:
            Registered data subject
        """
        jurisdiction = jurisdiction or self.default_jurisdiction
        
        subject = DataSubject(
            subject_id=subject_id,
            subject_type=subject_type,
            jurisdiction=jurisdiction
        )
        
        self.data_subjects[subject_id] = subject
        
        self.logger.info(f"Registered data subject: {subject_id} ({jurisdiction})")
        return subject
    
    def record_consent(
        self,
        subject_id: str,
        purposes: List[ProcessingPurpose],
        retention_days: int = 365
    ) -> bool:
        """Record consent from data subject.
        
        Args:
            subject_id: Subject identifier
            purposes: List of processing purposes
            retention_days: Data retention period
            
        Returns:
            True if consent recorded successfully
        """
        if subject_id not in self.data_subjects:
            self.logger.error(f"Unknown data subject: {subject_id}")
            return False
        
        subject = self.data_subjects[subject_id]
        subject.consent_date = time.time()
        subject.consent_purposes.update(purposes)
        subject.retention_period_days = retention_days
        
        self.logger.info(
            f"Consent recorded for {subject_id}: {[p.value for p in purposes]}"
        )
        return True
    
    def withdraw_consent(self, subject_id: str, purposes: List[ProcessingPurpose] = None):
        """Withdraw consent for data processing.
        
        Args:
            subject_id: Subject identifier
            purposes: Specific purposes to withdraw (all if None)
        """
        if subject_id not in self.data_subjects:
            self.logger.error(f"Unknown data subject: {subject_id}")
            return
        
        subject = self.data_subjects[subject_id]
        
        if purposes is None:
            # Withdraw all consent
            subject.consent_purposes.clear()
            self.logger.info(f"All consent withdrawn for {subject_id}")
        else:
            # Withdraw specific purposes
            for purpose in purposes:
                subject.consent_purposes.discard(purpose)
            self.logger.info(
                f"Consent withdrawn for {subject_id}: {[p.value for p in purposes]}"
            )
        
        # Mark related records for deletion
        self._schedule_data_deletion(subject_id, purposes)
    
    def record_processing(
        self,
        subject_id: str,
        data_type: str,
        purpose: ProcessingPurpose,
        classification: DataClassification = DataClassification.INTERNAL,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Record data processing activity.
        
        Args:
            subject_id: Subject identifier
            data_type: Type of data being processed
            purpose: Purpose of processing
            classification: Data classification level
            metadata: Additional metadata
            
        Returns:
            Record ID
        """
        # Check consent
        if subject_id in self.data_subjects:
            subject = self.data_subjects[subject_id]
            
            if not subject.has_valid_consent(purpose):
                self.logger.warning(
                    f"Processing without valid consent: {subject_id} for {purpose.value}"
                )
                
                # Check if processing is legally justified without consent
                if not self._is_processing_legally_justified(purpose, subject.jurisdiction):
                    raise ValueError(f"Processing not permitted: no consent for {purpose.value}")
        
        # Generate record
        record_id = self._generate_record_id(subject_id, data_type, purpose)
        
        subject = self.data_subjects.get(subject_id)
        jurisdiction = subject.jurisdiction if subject else self.default_jurisdiction
        retention_days = subject.retention_period_days if subject else 365
        
        record = DataRecord(
            record_id=record_id,
            subject_id=subject_id,
            data_type=data_type,
            classification=classification,
            purpose=purpose,
            processing_date=time.time(),
            retention_until=time.time() + (retention_days * 24 * 3600),
            jurisdiction=jurisdiction,
            metadata=metadata or {}
        )
        
        self.processing_records.append(record)
        
        self.logger.debug(f"Processing recorded: {record_id}")
        return record_id
    
    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Get all data for a subject (GDPR Article 15 - Right of access).
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            All data related to the subject
        """
        if subject_id not in self.data_subjects:
            return {}
        
        subject = self.data_subjects[subject_id]
        subject_records = [r for r in self.processing_records if r.subject_id == subject_id]
        
        return {
            'subject_info': {
                'subject_id': subject.subject_id,
                'subject_type': subject.subject_type,
                'jurisdiction': subject.jurisdiction,
                'consent_date': subject.consent_date,
                'consent_purposes': [p.value for p in subject.consent_purposes],
                'retention_period_days': subject.retention_period_days
            },
            'processing_records': [
                {
                    'record_id': r.record_id,
                    'data_type': r.data_type,
                    'classification': r.classification.value,
                    'purpose': r.purpose.value,
                    'processing_date': r.processing_date,
                    'retention_until': r.retention_until,
                    'metadata': r.metadata
                }
                for r in subject_records
            ],
            'retention_status': [
                {
                    'record_id': r.record_id,
                    'expired': r.is_expired(),
                    'days_until_deletion': max(0, int((r.retention_until - time.time()) / 86400))
                }
                for r in subject_records
            ]
        }
    
    def delete_subject_data(self, subject_id: str, verify: bool = True) -> Dict[str, Any]:
        """Delete all data for a subject (GDPR Article 17 - Right to erasure).
        
        Args:
            subject_id: Subject identifier
            verify: Whether to verify deletion
            
        Returns:
            Deletion report
        """
        if subject_id not in self.data_subjects:
            return {'status': 'error', 'message': 'Subject not found'}
        
        # Count records before deletion
        subject_records = [r for r in self.processing_records if r.subject_id == subject_id]
        records_count = len(subject_records)
        
        # Remove processing records
        self.processing_records = [
            r for r in self.processing_records if r.subject_id != subject_id
        ]
        
        # Remove subject registration
        del self.data_subjects[subject_id]
        
        # Verify deletion if requested
        verification_status = "not_verified"
        if verify:
            remaining_records = [r for r in self.processing_records if r.subject_id == subject_id]
            if len(remaining_records) == 0 and subject_id not in self.data_subjects:
                verification_status = "verified"
            else:
                verification_status = "failed"
        
        report = {
            'status': 'completed',
            'subject_id': subject_id,
            'records_deleted': records_count,
            'deletion_date': time.time(),
            'verification_status': verification_status
        }
        
        self.logger.info(f"Data deleted for subject {subject_id}: {records_count} records")
        return report
    
    def anonymize_expired_data(self) -> Dict[str, Any]:
        """Anonymize expired data records."""
        expired_records = [r for r in self.processing_records if r.is_expired()]
        anonymized_count = 0
        
        for record in expired_records:
            # Check if anonymization is appropriate
            if self._should_anonymize(record):
                # Replace with anonymized version
                anon_record = record.anonymize()
                
                # Remove original and add anonymized
                self.processing_records.remove(record)
                self.processing_records.append(anon_record)
                
                anonymized_count += 1
            else:
                # Delete entirely
                self.processing_records.remove(record)
        
        report = {
            'timestamp': time.time(),
            'expired_records': len(expired_records),
            'anonymized_records': anonymized_count,
            'deleted_records': len(expired_records) - anonymized_count
        }
        
        self.logger.info(f"Data cleanup: {anonymized_count} anonymized, "
                        f"{len(expired_records) - anonymized_count} deleted")
        
        return report
    
    def generate_compliance_report(self, jurisdiction: str = None) -> Dict[str, Any]:
        """Generate compliance report for audit purposes.
        
        Args:
            jurisdiction: Specific jurisdiction to report on
            
        Returns:
            Compliance report
        """
        target_jurisdictions = [jurisdiction] if jurisdiction else \
                             set(s.jurisdiction for s in self.data_subjects.values())
        
        report = {
            'organization': self.organization_name,
            'report_date': time.time(),
            'jurisdiction_reports': {}
        }
        
        for juris in target_jurisdictions:
            subjects = [s for s in self.data_subjects.values() if s.jurisdiction == juris]
            records = [r for r in self.processing_records if r.jurisdiction == juris]
            
            # Analyze consent status
            consent_stats = self._analyze_consent_status(subjects)
            
            # Analyze data retention
            retention_stats = self._analyze_retention_compliance(records)
            
            # Generate jurisdiction-specific compliance checks
            compliance_checks = self._run_compliance_checks(juris, subjects, records)
            
            report['jurisdiction_reports'][juris] = {
                'subjects_count': len(subjects),
                'records_count': len(records),
                'consent_statistics': consent_stats,
                'retention_statistics': retention_stats,
                'compliance_checks': compliance_checks
            }
        
        return report
    
    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance rules for different jurisdictions."""
        return {
            'EU': {  # GDPR
                'name': 'General Data Protection Regulation',
                'requires_consent': True,
                'max_retention_days': 365,
                'anonymization_required': True,
                'right_to_access': True,
                'right_to_erasure': True,
                'right_to_portability': True,
                'breach_notification_hours': 72,
                'legal_bases': [
                    'consent', 'contract', 'legal_obligation', 
                    'vital_interests', 'public_task', 'legitimate_interests'
                ]
            },
            'US-CA': {  # CCPA
                'name': 'California Consumer Privacy Act',
                'requires_consent': False,  # Opt-out model
                'max_retention_days': None,
                'anonymization_required': False,
                'right_to_access': True,
                'right_to_erasure': True,
                'right_to_portability': True,
                'sale_opt_out': True
            },
            'SG': {  # PDPA
                'name': 'Personal Data Protection Act',
                'requires_consent': True,
                'max_retention_days': None,
                'anonymization_required': False,
                'right_to_access': True,
                'right_to_erasure': False,  # Limited
                'notification_required': True
            }
        }
    
    def _is_processing_legally_justified(
        self, 
        purpose: ProcessingPurpose, 
        jurisdiction: str
    ) -> bool:
        """Check if processing is legally justified without explicit consent."""
        rules = self.compliance_rules.get(jurisdiction, {})
        
        # Some purposes may be legally justified
        justified_purposes = {
            ProcessingPurpose.DEBUGGING: True,  # System operation
            ProcessingPurpose.MONITORING: True,  # Security monitoring
        }
        
        return justified_purposes.get(purpose, False)
    
    def _generate_record_id(
        self, 
        subject_id: str, 
        data_type: str, 
        purpose: ProcessingPurpose
    ) -> str:
        """Generate unique record ID."""
        timestamp = str(int(time.time() * 1000))
        content = f"{subject_id}:{data_type}:{purpose.value}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _schedule_data_deletion(
        self,
        subject_id: str,
        purposes: List[ProcessingPurpose] = None
    ):
        """Schedule data for deletion based on consent withdrawal."""
        # In production, this would schedule background deletion
        records_to_delete = []
        
        for record in self.processing_records:
            if record.subject_id == subject_id:
                if purposes is None or record.purpose in purposes:
                    records_to_delete.append(record)
        
        for record in records_to_delete:
            self.processing_records.remove(record)
        
        self.logger.info(f"Scheduled {len(records_to_delete)} records for deletion")
    
    def _should_anonymize(self, record: DataRecord) -> bool:
        """Determine if record should be anonymized vs deleted."""
        # Anonymize if has research/analytics value and classification allows
        research_purposes = {ProcessingPurpose.RESEARCH, ProcessingPurpose.ANALYTICS}
        
        return (record.purpose in research_purposes and 
                record.classification in {DataClassification.PUBLIC, DataClassification.INTERNAL})
    
    def _analyze_consent_status(self, subjects: List[DataSubject]) -> Dict[str, Any]:
        """Analyze consent status for subjects."""
        total_subjects = len(subjects)
        
        if total_subjects == 0:
            return {'total': 0, 'with_consent': 0, 'consent_rate': 0.0}
        
        with_consent = sum(1 for s in subjects if s.consent_purposes)
        
        return {
            'total': total_subjects,
            'with_consent': with_consent,
            'consent_rate': with_consent / total_subjects,
            'average_purposes': sum(len(s.consent_purposes) for s in subjects) / total_subjects
        }
    
    def _analyze_retention_compliance(self, records: List[DataRecord]) -> Dict[str, Any]:
        """Analyze data retention compliance."""
        total_records = len(records)
        
        if total_records == 0:
            return {'total': 0, 'expired': 0, 'compliance_rate': 1.0}
        
        expired_records = sum(1 for r in records if r.is_expired())
        
        return {
            'total': total_records,
            'expired': expired_records,
            'compliance_rate': 1.0 - (expired_records / total_records),
            'avg_retention_days': sum(
                (r.retention_until - r.processing_date) / 86400 for r in records
            ) / total_records
        }
    
    def _run_compliance_checks(
        self,
        jurisdiction: str,
        subjects: List[DataSubject],
        records: List[DataRecord]
    ) -> Dict[str, Any]:
        """Run jurisdiction-specific compliance checks."""
        rules = self.compliance_rules.get(jurisdiction, {})
        checks = {}
        
        # Check consent requirement compliance
        if rules.get('requires_consent', False):
            subjects_without_consent = [s for s in subjects if not s.consent_purposes]
            checks['consent_compliance'] = {
                'required': True,
                'violations': len(subjects_without_consent),
                'compliant': len(subjects_without_consent) == 0
            }
        
        # Check retention period compliance
        if 'max_retention_days' in rules and rules['max_retention_days']:
            max_days = rules['max_retention_days']
            violations = []
            
            for record in records:
                retention_days = (record.retention_until - record.processing_date) / 86400
                if retention_days > max_days:
                    violations.append(record.record_id)
            
            checks['retention_compliance'] = {
                'max_allowed_days': max_days,
                'violations': len(violations),
                'compliant': len(violations) == 0
            }
        
        # Check expired data cleanup
        expired_records = [r for r in records if r.is_expired()]
        checks['data_cleanup'] = {
            'expired_records': len(expired_records),
            'requires_action': len(expired_records) > 0
        }
        
        return checks
    
    def export_data_inventory(self, file_path: Path) -> bool:
        """Export data inventory for audit purposes.
        
        Args:
            file_path: Path to save inventory
            
        Returns:
            True if export successful
        """
        try:
            inventory = {
                'organization': self.organization_name,
                'export_date': time.time(),
                'data_subjects': [
                    {
                        'subject_id': s.subject_id,
                        'subject_type': s.subject_type,
                        'jurisdiction': s.jurisdiction,
                        'consent_date': s.consent_date,
                        'consent_purposes': [p.value for p in s.consent_purposes],
                        'retention_period_days': s.retention_period_days
                    }
                    for s in self.data_subjects.values()
                ],
                'processing_records': [
                    {
                        'record_id': r.record_id,
                        'subject_id': r.subject_id,
                        'data_type': r.data_type,
                        'classification': r.classification.value,
                        'purpose': r.purpose.value,
                        'processing_date': r.processing_date,
                        'retention_until': r.retention_until,
                        'jurisdiction': r.jurisdiction,
                        'expired': r.is_expired()
                    }
                    for r in self.processing_records
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(inventory, f, indent=2)
            
            self.logger.info(f"Data inventory exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export data inventory: {e}")
            return False