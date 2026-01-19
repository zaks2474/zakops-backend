#!/usr/bin/env python3
"""
Deal Registry Module - Canonical source of truth for deal identification

This module provides a robust deal matching system that replaces weak text-based
matching with a multi-tier alias system supporting:
- Listing ID matching (exact)
- Email thread continuity
- Company name matching (exact and fuzzy)
- Keyword combination matching
- Broker + sector heuristics
"""

import json
import re
import csv
import sqlite3
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("rapidfuzz not available, fuzzy matching disabled")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Alias:
    """An alias that can match to a deal"""
    alias: str
    alias_normalized: str
    alias_type: str  # listing_id, listing_number, company_name, dba_name, name_variation, subject_keywords, location_sector, broker_thread, email_hash
    confidence: float = 1.0
    source: str = "manual"  # manual, email_extraction, auto_generated, migration
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Identifiers:
    """Deal identifiers from various sources"""
    listing_ids: List[str] = field(default_factory=list)
    broker_reference_ids: List[str] = field(default_factory=list)
    bizbuysell_id: Optional[str] = None
    axial_id: Optional[str] = None
    internal_codes: List[str] = field(default_factory=list)


@dataclass
class Location:
    """Deal location information"""
    city: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None


@dataclass
class CompanyInfo:
    """Company information for a deal"""
    company_name: Optional[str] = None
    legal_entity: Optional[str] = None
    dba_names: List[str] = field(default_factory=list)
    location: Optional[Location] = None
    sector: Optional[str] = None
    franchise_system: Optional[str] = None


@dataclass
class BrokerInfo:
    """Broker information"""
    broker_id: Optional[str] = None
    name: str = ""
    email: str = ""
    phone: str = ""
    company: str = ""
    quality_rating: str = "MEDIUM"
    sectors: List[str] = field(default_factory=list)
    domain: Optional[str] = None  # Email domain for broker identification


@dataclass
class DealMetadata:
    """Additional deal metadata"""
    priority: str = "MEDIUM"
    asking_price: Optional[str] = None
    ebitda: Optional[str] = None
    revenue: Optional[str] = None
    employees: Optional[int] = None
    nda_status: str = "none"  # none, pending, signed
    cim_received: bool = False
    junk_reason: Optional[str] = None
    archived_at: Optional[str] = None


@dataclass
class AuditEntry:
    """Audit trail entry"""
    timestamp: str
    action: str  # created, alias_added, folder_merged, stage_changed, archived
    source: str  # manual, email_sync, cli_command, migration
    details: str


@dataclass
class Deal:
    """Complete deal record"""
    deal_id: str
    canonical_name: str
    display_name: Optional[str] = None
    folder_path: Optional[str] = None
    stage: str = "inbound"  # inbound, screening, qualified, loi, closing, archive, rejected
    status: str = "active"  # active, inactive, merged, archived, junk
    identifiers: Identifiers = field(default_factory=Identifiers)
    company_info: CompanyInfo = field(default_factory=CompanyInfo)
    broker: Optional[BrokerInfo] = None
    aliases: List[Alias] = field(default_factory=list)
    email_thread_ids: List[str] = field(default_factory=list)
    related_folders: List[str] = field(default_factory=list)
    metadata: DealMetadata = field(default_factory=DealMetadata)
    audit_trail: List[AuditEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    deleted: bool = False
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None
    deleted_reason: Optional[str] = None

    def add_alias(self, alias: str, alias_type: str, confidence: float = 1.0, source: str = "manual"):
        """Add an alias to the deal"""
        normalized = normalize_text(alias)
        # Check for duplicate
        for existing in self.aliases:
            if existing.alias_normalized == normalized and existing.alias_type == alias_type:
                return False
        self.aliases.append(Alias(
            alias=alias,
            alias_normalized=normalized,
            alias_type=alias_type,
            confidence=confidence,
            source=source
        ))
        self.updated_at = datetime.now().isoformat()
        return True

    def add_audit(self, action: str, source: str, details: str):
        """Add audit trail entry"""
        self.audit_trail.append(AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=action,
            source=source,
            details=details
        ))


@dataclass
class JunkPattern:
    """Pattern for identifying junk emails"""
    pattern: str
    pattern_type: str  # domain, subject_regex, subject_contains, sender_email
    action: str = "reject"
    notes: str = ""


@dataclass
class MatchResult:
    """Result of a matching attempt"""
    matched: bool
    deal_id: Optional[str] = None
    confidence: float = 0.0
    match_type: Optional[str] = None
    matched_alias: Optional[str] = None
    reason: Optional[str] = None
    action: str = "none"  # none, create_new, reject
    suggested_aliases: List[Tuple[str, str]] = field(default_factory=list)  # (alias, type)


@dataclass
class EmailContent:
    """Email content for matching"""
    subject: str
    body: str
    sender: str
    message_id: Optional[str] = None
    thread_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    received_date: Optional[str] = None


@dataclass
class ExtractedIdentifiers:
    """Identifiers extracted from email content"""
    listing_ids: List[str] = field(default_factory=list)
    broker_email: Optional[str] = None
    broker_domain: Optional[str] = None
    company_names: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching"""
    if not text:
        return ""
    # Lowercase, strip, remove extra whitespace
    result = text.lower().strip()
    result = re.sub(r'\s+', ' ', result)
    # Remove common punctuation
    result = re.sub(r'[^\w\s-]', '', result)
    return result


def normalize_company_name(name: str) -> str:
    """Normalize company name for matching"""
    if not name:
        return ""
    result = normalize_text(name)
    # Remove common suffixes
    suffixes = ['llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'co', 'company']
    for suffix in suffixes:
        result = re.sub(rf'\b{suffix}\b', '', result)
    return result.strip()


def extract_email_address(sender: str) -> str:
    """Extract email address from sender string"""
    match = re.search(r'<([^>]+)>', sender)
    if match:
        return match.group(1).lower().strip()
    # If no angle brackets, assume entire string is email
    if '@' in sender:
        return sender.lower().strip()
    return ""


def extract_domain(email: str) -> str:
    """Extract domain from email address"""
    addr = extract_email_address(email)
    if '@' in addr:
        return addr.split('@')[1]
    return ""


# ============================================================================
# Identifier Extractor
# ============================================================================

class IdentifierExtractor:
    """Extract deal identifiers from email content"""

    # Listing ID patterns
    LISTING_PATTERNS = [
        r'listing\s*[#:]?\s*(\d{4,8})',           # "Listing #57839", "Listing: 57839"
        r'listing\s+id[:\s]*(\d{4,8})',           # "Listing ID: 57839"
        r'ref(?:erence)?[#:\s]*(\d{4,8})',        # "Ref #57839"
        r'deal\s*[#:\s]*(\d{4,8})',               # "Deal #12345"
        r'\babb(\d{5})\b',                        # "ABB25034" (Alpine Business Brokers)
        r'\b(\d{5,8}-\d{5,8})\b',                 # "8455-403820" patterns
        r'\b(\d{7})\b',                           # Standalone 7-digit numbers (BizBuySell)
    ]

    # Sector patterns
    SECTOR_PATTERNS = [
        (r'\bmsp\b', 'MSP'),
        (r'\bmanaged service', 'MSP'),
        (r'\bit service', 'IT-Services'),
        (r'\bsaas\b', 'SaaS'),
        (r'\becommerce\b', 'Ecommerce'),
        (r'\be-commerce\b', 'Ecommerce'),
        (r'\bsoftware\b', 'Software'),
        (r'\bcybersecurity\b', 'Cybersecurity'),
        (r'\bdigital marketing\b', 'Digital-Marketing'),
        (r'\bfba\b', 'FBA'),
        (r'\bamazon\b', 'Amazon'),
    ]

    # Location patterns
    LOCATION_PATTERNS = [
        (r'\btexas\b', 'Texas'),
        (r'\btx\b', 'Texas'),
        (r'\bcalifornia\b', 'California'),
        (r'\bca\b', 'California'),
        (r'\bflorida\b', 'Florida'),
        (r'\bfl\b', 'Florida'),
        (r'\bnew york\b', 'New York'),
        (r'\bny\b', 'New York'),
        (r'\bdallas\b', 'Dallas'),
        (r'\bdenver\b', 'Denver'),
        (r'\bhouston\b', 'Houston'),
    ]

    # Stop words for keyword extraction
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'new', 'listing', 'deal', 'opportunity', 'business', 'company',
        'email', 'please', 'thank', 'thanks', 'regards', 'best', 'hello', 'hi',
        'attached', 'attachment', 'information', 'details', 'following', 'review',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'here', 'there', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'whose', 'with', 'without',
        'from', 'into', 'onto', 'upon', 'about', 'above', 'below', 'between',
        'under', 'over', 'through', 'during', 'before', 'after', 'since', 'until',
        'while', 'for', 'your', 'our', 'their', 'its', 'his', 'her', 'my', 'me',
        'you', 'him', 'her', 'them', 'we', 'they', 'who', 'one', 'two', 'three',
    }

    def extract(self, content: EmailContent) -> ExtractedIdentifiers:
        """Extract all potential identifiers from email content"""
        text = f"{content.subject} {content.body[:3000]}".lower()
        identifiers = ExtractedIdentifiers()

        # Extract listing IDs
        for pattern in self.LISTING_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in identifiers.listing_ids:
                    identifiers.listing_ids.append(match)

        # Extract broker info
        identifiers.broker_email = extract_email_address(content.sender)
        identifiers.broker_domain = extract_domain(content.sender)

        # Extract sectors
        for pattern, sector in self.SECTOR_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if sector not in identifiers.sectors:
                    identifiers.sectors.append(sector)

        # Extract locations
        for pattern, location in self.LOCATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if location not in identifiers.locations:
                    identifiers.locations.append(location)

        # Extract significant keywords
        identifiers.keywords = self._extract_keywords(text)

        # Extract company names (capitalized phrases)
        identifiers.company_names = self._extract_company_names(content.subject)

        return identifiers

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords"""
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        significant = [w for w in words if w not in self.STOP_WORDS]

        # Count occurrences
        from collections import Counter
        word_counts = Counter(significant)
        return [w for w, c in word_counts.most_common(20)]

    def _extract_company_names(self, subject: str) -> List[str]:
        """Extract potential company names from subject"""
        names = []
        # Look for capitalized word sequences
        matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', subject)
        for match in matches:
            if len(match) > 3 and match.lower() not in self.STOP_WORDS:
                names.append(match)
        return names


# ============================================================================
# Deal Matcher
# ============================================================================

class DealMatcher:
    """Multi-tier matching system for incoming content against Deal Registry"""

    MATCH_TIERS = {
        'listing_id_exact': {'confidence': 1.0, 'tier': 1},
        'broker_reference': {'confidence': 0.98, 'tier': 2},
        'email_thread': {'confidence': 0.95, 'tier': 3},
        'broker_listing_combo': {'confidence': 0.95, 'tier': 4},
        'company_name_exact': {'confidence': 0.90, 'tier': 5},
        'company_name_fuzzy': {'confidence': 0.85, 'tier': 6},
        'keyword_match': {'confidence': 0.70, 'tier': 7},
        'broker_sector_location': {'confidence': 0.60, 'tier': 8},
    }

    def __init__(self, registry: 'DealRegistry'):
        self.registry = registry
        self.extractor = IdentifierExtractor()

    def match(self, content: EmailContent) -> MatchResult:
        """Attempt to match content against registry"""
        # Step 1: Check junk patterns first
        junk_result = self.check_junk(content)
        if junk_result:
            return MatchResult(
                matched=False,
                reason=f'junk_pattern: {junk_result}',
                action='reject'
            )

        # Step 2: Extract identifiers from content
        identifiers = self.extractor.extract(content)

        # Step 3: Try each matching tier
        result = self._try_listing_id_match(identifiers)
        if result and result.matched:
            return result

        result = self._try_email_thread_match(content)
        if result and result.matched:
            return result

        result = self._try_broker_listing_combo(identifiers)
        if result and result.matched:
            return result

        result = self._try_company_name_exact(identifiers)
        if result and result.matched:
            return result

        if FUZZY_AVAILABLE:
            result = self._try_company_name_fuzzy(identifiers)
            if result and result.matched:
                return result

        result = self._try_keyword_match(identifiers)
        if result and result.matched:
            return result

        result = self._try_broker_sector_location(identifiers)
        if result and result.matched:
            return result

        # No match found - suggest creating new deal
        suggested_aliases = []
        for lid in identifiers.listing_ids:
            suggested_aliases.append((lid, 'listing_number'))
        for company in identifiers.company_names:
            suggested_aliases.append((company, 'company_name'))

        return MatchResult(
            matched=False,
            reason='no_match',
            action='create_new',
            suggested_aliases=suggested_aliases
        )

    def check_junk(self, content: EmailContent) -> Optional[str]:
        """Check if content matches junk patterns"""
        for pattern in self.registry.junk_patterns:
            if pattern.pattern_type == 'domain':
                sender_domain = extract_domain(content.sender)
                if pattern.pattern.lower() in sender_domain.lower():
                    return pattern.pattern
            elif pattern.pattern_type == 'subject_regex':
                if re.search(pattern.pattern, content.subject, re.IGNORECASE):
                    return pattern.pattern
            elif pattern.pattern_type == 'subject_contains':
                if pattern.pattern.lower() in content.subject.lower():
                    return pattern.pattern
            elif pattern.pattern_type == 'sender_email':
                sender_email = extract_email_address(content.sender)
                if pattern.pattern.lower() in sender_email.lower():
                    return pattern.pattern
        return None

    def _try_listing_id_match(self, identifiers: ExtractedIdentifiers) -> Optional[MatchResult]:
        """Tier 1: Exact listing ID match"""
        for listing_id in identifiers.listing_ids:
            for deal_id, deal in self.registry.deals.items():
                if deal.status not in ('active', 'inactive'):
                    continue
                # Check identifiers
                if listing_id in deal.identifiers.listing_ids:
                    return MatchResult(
                        matched=True,
                        deal_id=deal_id,
                        confidence=1.0,
                        match_type='listing_id_exact',
                        matched_alias=listing_id
                    )
                # Check aliases
                for alias in deal.aliases:
                    if alias.alias_type in ('listing_id', 'listing_number'):
                        if listing_id == alias.alias_normalized or listing_id in alias.alias_normalized:
                            return MatchResult(
                                matched=True,
                                deal_id=deal_id,
                                confidence=alias.confidence,
                                match_type='listing_id_exact',
                                matched_alias=alias.alias
                            )
        return None

    def _try_email_thread_match(self, content: EmailContent) -> Optional[MatchResult]:
        """Tier 3: Email thread continuity"""
        if content.thread_id:
            for deal_id, deal in self.registry.deals.items():
                if deal.status not in ('active', 'inactive'):
                    continue
                if content.thread_id in deal.email_thread_ids:
                    return MatchResult(
                        matched=True,
                        deal_id=deal_id,
                        confidence=0.95,
                        match_type='email_thread',
                        matched_alias=f"thread:{content.thread_id}"
                    )

        if content.in_reply_to:
            # Check email-to-deal mappings
            mapping = self.registry.get_email_deal_mapping(content.in_reply_to)
            if mapping:
                return MatchResult(
                    matched=True,
                    deal_id=mapping,
                    confidence=0.95,
                    match_type='email_thread',
                    matched_alias=f"reply_to:{content.in_reply_to}"
                )
        return None

    def _try_broker_listing_combo(self, identifiers: ExtractedIdentifiers) -> Optional[MatchResult]:
        """Tier 4: Broker + listing ID combination"""
        if not identifiers.broker_email or not identifiers.listing_ids:
            return None

        broker_domain = identifiers.broker_domain

        for deal_id, deal in self.registry.deals.items():
            if deal.status not in ('active', 'inactive'):
                continue
            if not deal.broker:
                continue

            # Check if same broker domain
            deal_broker_domain = extract_domain(deal.broker.email)
            if broker_domain != deal_broker_domain:
                continue

            # Check for listing ID match
            for listing_id in identifiers.listing_ids:
                if listing_id in deal.identifiers.listing_ids:
                    return MatchResult(
                        matched=True,
                        deal_id=deal_id,
                        confidence=0.95,
                        match_type='broker_listing_combo',
                        matched_alias=f"{identifiers.broker_email}+{listing_id}"
                    )
        return None

    def _try_company_name_exact(self, identifiers: ExtractedIdentifiers) -> Optional[MatchResult]:
        """Tier 5: Exact company name match"""
        for company in identifiers.company_names:
            normalized = normalize_company_name(company)
            if len(normalized) < 4:
                continue

            for deal_id, deal in self.registry.deals.items():
                if deal.status not in ('active', 'inactive'):
                    continue

                # Check company_info
                if deal.company_info.company_name:
                    deal_normalized = normalize_company_name(deal.company_info.company_name)
                    if normalized == deal_normalized:
                        return MatchResult(
                            matched=True,
                            deal_id=deal_id,
                            confidence=0.90,
                            match_type='company_name_exact',
                            matched_alias=company
                        )

                # Check aliases
                for alias in deal.aliases:
                    if alias.alias_type in ('company_name', 'dba_name', 'name_variation'):
                        if normalized == alias.alias_normalized:
                            return MatchResult(
                                matched=True,
                                deal_id=deal_id,
                                confidence=0.90,
                                match_type='company_name_exact',
                                matched_alias=alias.alias
                            )
        return None

    def _try_company_name_fuzzy(self, identifiers: ExtractedIdentifiers) -> Optional[MatchResult]:
        """Tier 6: Fuzzy company name matching"""
        if not FUZZY_AVAILABLE:
            return None

        best_match = None
        best_score = 0

        for company in identifiers.company_names:
            normalized = normalize_company_name(company)
            if len(normalized) < 4:
                continue

            for deal_id, deal in self.registry.deals.items():
                if deal.status not in ('active', 'inactive'):
                    continue

                # Check aliases
                for alias in deal.aliases:
                    if alias.alias_type in ('company_name', 'dba_name', 'name_variation'):
                        score = fuzz.token_sort_ratio(normalized, alias.alias_normalized)
                        if score >= 85 and score > best_score:
                            best_score = score
                            best_match = MatchResult(
                                matched=True,
                                deal_id=deal_id,
                                confidence=score / 100,
                                match_type='company_name_fuzzy',
                                matched_alias=alias.alias
                            )
        return best_match

    def _try_keyword_match(self, identifiers: ExtractedIdentifiers) -> Optional[MatchResult]:
        """Tier 7: Keyword combination matching (improved from 2+ word match)"""
        MIN_KEYWORD_MATCHES = 3
        MIN_MATCH_RATIO = 0.4

        best_match = None
        best_score = 0

        incoming_keywords = set(identifiers.keywords)
        if len(incoming_keywords) < MIN_KEYWORD_MATCHES:
            return None

        for deal_id, deal in self.registry.deals.items():
            if deal.status not in ('active', 'inactive'):
                continue

            # Collect deal keywords from aliases and canonical name
            deal_keywords = set()
            for alias in deal.aliases:
                if alias.alias_type == 'subject_keywords':
                    deal_keywords.update(alias.alias_normalized.split())

            # Also include canonical name words
            deal_keywords.update(normalize_text(deal.canonical_name).split())

            if not deal_keywords:
                continue

            # Calculate overlap
            overlap = deal_keywords & incoming_keywords
            if len(overlap) >= MIN_KEYWORD_MATCHES:
                ratio = len(overlap) / max(len(deal_keywords), len(incoming_keywords))
                score = len(overlap) * ratio

                if ratio >= MIN_MATCH_RATIO and score > best_score:
                    best_score = score
                    best_match = MatchResult(
                        matched=True,
                        deal_id=deal_id,
                        confidence=min(0.7, 0.5 + (ratio * 0.3)),
                        match_type='keyword_match',
                        matched_alias=f"keywords:{','.join(list(overlap)[:5])}"
                    )

        return best_match

    def _try_broker_sector_location(self, identifiers: ExtractedIdentifiers) -> Optional[MatchResult]:
        """Tier 8: Broker + sector + location heuristic"""
        if not identifiers.broker_email:
            return None
        if not identifiers.sectors and not identifiers.locations:
            return None

        broker_domain = identifiers.broker_domain

        for deal_id, deal in self.registry.deals.items():
            if deal.status not in ('active', 'inactive'):
                continue
            if not deal.broker:
                continue

            # Check same broker
            deal_broker_domain = extract_domain(deal.broker.email)
            if broker_domain != deal_broker_domain:
                continue

            # Check sector match
            sector_match = False
            if identifiers.sectors and deal.company_info.sector:
                for sector in identifiers.sectors:
                    if sector.lower() in deal.company_info.sector.lower():
                        sector_match = True
                        break

            # Check location match
            location_match = False
            if identifiers.locations and deal.company_info.location:
                for loc in identifiers.locations:
                    loc_lower = loc.lower()
                    if deal.company_info.location.state and loc_lower in deal.company_info.location.state.lower():
                        location_match = True
                        break
                    if deal.company_info.location.city and loc_lower in deal.company_info.location.city.lower():
                        location_match = True
                        break

            if sector_match and location_match:
                return MatchResult(
                    matched=True,
                    deal_id=deal_id,
                    confidence=0.60,
                    match_type='broker_sector_location',
                    matched_alias=f"{broker_domain}+{identifiers.sectors[0] if identifiers.sectors else ''}+{identifiers.locations[0] if identifiers.locations else ''}"
                )

        return None


# ============================================================================
# Deal Registry
# ============================================================================

class DealRegistry:
    """Canonical registry for all deals"""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.db_path = self.registry_path.parent / "deal_registry.db"
        self.deals: Dict[str, Deal] = {}
        self.brokers: Dict[str, BrokerInfo] = {}
        self.junk_patterns: List[JunkPattern] = []
        self.email_to_deal: Dict[str, str] = {}  # message_id -> deal_id
        self.thread_to_deal: Dict[str, str] = {}  # gmail_thread_id -> deal_id
        self.thread_to_non_deal: Dict[str, str] = {}  # gmail_thread_id -> rejection reason
        self._deal_counter = 0
        self._broker_counter = 0

        self._load()

    def _load(self):
        """Load registry from JSON file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._deserialize(data)
                logger.info(f"Loaded {len(self.deals)} deals from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.deals = {}
        else:
            logger.info("No existing registry found, starting fresh")

    def _deserialize(self, data: dict):
        """Deserialize registry data"""
        self._deal_counter = data.get('deal_counter', 0)
        self._broker_counter = data.get('broker_counter', 0)

        # Load deals
        for deal_id, deal_data in data.get('deals', {}).items():
            try:
                # Reconstruct nested objects
                identifiers = Identifiers(**deal_data.get('identifiers', {}))

                location_data = deal_data.get('company_info', {}).get('location')
                location = Location(**location_data) if location_data else None

                company_info_data = deal_data.get('company_info', {})
                company_info_data['location'] = location
                company_info = CompanyInfo(**{k: v for k, v in company_info_data.items() if k != 'location' or v is not None})
                company_info.location = location

                broker_data = deal_data.get('broker')
                broker = BrokerInfo(**broker_data) if broker_data else None

                metadata = DealMetadata(**deal_data.get('metadata', {}))

                aliases = [Alias(**a) for a in deal_data.get('aliases', [])]
                audit_trail = [AuditEntry(**e) for e in deal_data.get('audit_trail', [])]

                deal = Deal(
                    deal_id=deal_id,
                    canonical_name=deal_data.get('canonical_name', ''),
                    display_name=deal_data.get('display_name'),
                    folder_path=deal_data.get('folder_path'),
                    stage=deal_data.get('stage', 'inbound'),
                    status=deal_data.get('status', 'active'),
                    identifiers=identifiers,
                    company_info=company_info,
                    broker=broker,
                    aliases=aliases,
                    email_thread_ids=deal_data.get('email_thread_ids', []),
                    related_folders=deal_data.get('related_folders', []),
                    metadata=metadata,
                    audit_trail=audit_trail,
                    created_at=deal_data.get('created_at', datetime.now().isoformat()),
                    updated_at=deal_data.get('updated_at', datetime.now().isoformat()),
                    deleted=bool(deal_data.get('deleted', False)),
                    deleted_at=deal_data.get('deleted_at'),
                    deleted_by=deal_data.get('deleted_by'),
                    deleted_reason=deal_data.get('deleted_reason'),
                )
                self.deals[deal_id] = deal
            except Exception as e:
                logger.error(f"Failed to deserialize deal {deal_id}: {e}")

        # Load brokers
        for broker_id, broker_data in data.get('brokers', {}).items():
            try:
                self.brokers[broker_id] = BrokerInfo(**broker_data)
            except Exception as e:
                logger.error(f"Failed to deserialize broker {broker_id}: {e}")

        # Load junk patterns
        for pattern_data in data.get('junk_patterns', []):
            try:
                self.junk_patterns.append(JunkPattern(**pattern_data))
            except Exception as e:
                logger.error(f"Failed to deserialize junk pattern: {e}")

        # Load email mappings
        self.email_to_deal = data.get('email_to_deal', {})
        self.thread_to_deal = data.get('thread_to_deal', {})
        self.thread_to_non_deal = data.get('thread_to_non_deal', {})

    def _serialize(self) -> dict:
        """Serialize registry to dict"""
        deals_data = {}
        for deal_id, deal in self.deals.items():
            deal_dict = {
                'deal_id': deal.deal_id,
                'canonical_name': deal.canonical_name,
                'display_name': deal.display_name,
                'folder_path': deal.folder_path,
                'stage': deal.stage,
                'status': deal.status,
                'identifiers': asdict(deal.identifiers),
                'company_info': {
                    'company_name': deal.company_info.company_name,
                    'legal_entity': deal.company_info.legal_entity,
                    'dba_names': deal.company_info.dba_names,
                    'location': asdict(deal.company_info.location) if deal.company_info.location else None,
                    'sector': deal.company_info.sector,
                    'franchise_system': deal.company_info.franchise_system,
                },
                'broker': asdict(deal.broker) if deal.broker else None,
                'aliases': [asdict(a) for a in deal.aliases],
                'email_thread_ids': deal.email_thread_ids,
                'related_folders': deal.related_folders,
                'metadata': asdict(deal.metadata),
                'audit_trail': [asdict(e) for e in deal.audit_trail],
                'created_at': deal.created_at,
                'updated_at': deal.updated_at,
                'deleted': bool(deal.deleted),
                'deleted_at': deal.deleted_at,
                'deleted_by': deal.deleted_by,
                'deleted_reason': deal.deleted_reason,
            }
            deals_data[deal_id] = deal_dict

        return {
            'schema_version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'deal_counter': self._deal_counter,
            'broker_counter': self._broker_counter,
            'deals': deals_data,
            'brokers': {k: asdict(v) for k, v in self.brokers.items()},
            'junk_patterns': [asdict(p) for p in self.junk_patterns],
            'email_to_deal': self.email_to_deal,
            'thread_to_deal': self.thread_to_deal,
            'thread_to_non_deal': self.thread_to_non_deal,
        }

    def save(self):
        """Save registry to JSON file"""
        # Create backup first
        if self.registry_path.exists():
            backup_dir = self.registry_path.parent / "registry_backup"
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f"deal_registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(self.registry_path, backup_path)
            # Keep only last 10 backups
            backups = sorted(backup_dir.glob("deal_registry_*.json"))
            for old_backup in backups[:-10]:
                old_backup.unlink()

        # Ensure directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._serialize(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.deals)} deals to registry")

    def generate_deal_id(self) -> str:
        """Generate next deal ID"""
        self._deal_counter += 1
        year = datetime.now().year
        return f"DEAL-{year}-{self._deal_counter:03d}"

    def generate_broker_id(self) -> str:
        """Generate next broker ID"""
        self._broker_counter += 1
        return f"BRK-{self._broker_counter:03d}"

    def get_deal(self, deal_id: str) -> Optional[Deal]:
        """Get deal by ID"""
        return self.deals.get(deal_id)

    def get_deal_by_folder(self, folder_path: str) -> Optional[Deal]:
        """Get deal by folder path"""
        for deal in self.deals.values():
            if deal.folder_path == folder_path:
                return deal
        return None

    def get_email_deal_mapping(self, message_id: str) -> Optional[str]:
        """Get deal_id for a message_id"""
        return self.email_to_deal.get(message_id)

    def add_email_deal_mapping(self, message_id: str, deal_id: str):
        """Add email to deal mapping"""
        self.email_to_deal[message_id] = deal_id

    def get_thread_deal_mapping(self, thread_id: str) -> Optional[str]:
        """Get deal_id for a gmail thread_id."""
        return self.thread_to_deal.get(thread_id)

    def add_thread_deal_mapping(self, thread_id: str, deal_id: str) -> None:
        """Map a gmail thread_id to a deal_id (overwrites existing mapping)."""
        if not thread_id:
            return
        self.thread_to_deal[thread_id] = deal_id

    def get_thread_non_deal_mapping(self, thread_id: str) -> Optional[str]:
        """Get rejection reason if thread was marked as non-deal."""
        return self.thread_to_non_deal.get(thread_id)

    def add_thread_non_deal_mapping(self, thread_id: str, reason: str) -> None:
        """Mark a gmail thread_id as non-deal (rejected)."""
        if not thread_id:
            return
        self.thread_to_non_deal[thread_id] = (reason or "").strip()[:500] or "rejected"

    def is_thread_resolved(self, thread_id: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if a gmail thread has been resolved deterministically.

        Returns: (is_resolved, deal_id_or_none, non_deal_reason_or_none)
        """
        tid = (thread_id or "").strip()
        if not tid:
            return (False, None, None)
        if tid in self.thread_to_deal:
            return (True, self.thread_to_deal.get(tid), None)
        if tid in self.thread_to_non_deal:
            return (True, None, self.thread_to_non_deal.get(tid))
        return (False, None, None)

    def create_deal(
        self,
        deal_id: str,
        canonical_name: str,
        folder_path: str,
        broker: Optional[BrokerInfo] = None,
        source: str = "email_sync"
    ) -> Deal:
        """Create a new deal in the registry"""
        deal = Deal(
            deal_id=deal_id,
            canonical_name=canonical_name,
            folder_path=folder_path,
            broker=broker,
        )
        deal.add_audit("created", source, f"Deal created: {canonical_name}")

        # Auto-generate aliases from name
        normalized_name = normalize_text(canonical_name)
        deal.add_alias(normalized_name, "name_variation", 1.0, "auto_generated")

        self.deals[deal_id] = deal
        return deal

    def add_alias(self, deal_id: str, alias: str, alias_type: str,
                  confidence: float = 1.0, source: str = "manual") -> bool:
        """Add alias to existing deal"""
        deal = self.deals.get(deal_id)
        if not deal:
            return False

        added = deal.add_alias(alias, alias_type, confidence, source)
        if added:
            deal.add_audit("alias_added", source, f"Added alias: {alias} ({alias_type})")
        return added

    def add_junk_pattern(self, pattern: str, pattern_type: str,
                         action: str = "reject", notes: str = "") -> bool:
        """Add a junk pattern"""
        # Check for duplicate
        for existing in self.junk_patterns:
            if existing.pattern == pattern and existing.pattern_type == pattern_type:
                return False

        self.junk_patterns.append(JunkPattern(
            pattern=pattern,
            pattern_type=pattern_type,
            action=action,
            notes=notes
        ))
        return True

    def archive_deal(self, deal_id: str, reason: str = "", source: str = "manual"):
        """Archive a deal"""
        deal = self.deals.get(deal_id)
        if deal:
            deal.status = "archived"
            deal.stage = "archive"
            deal.metadata.archived_at = datetime.now().isoformat()
            deal.add_audit("archived", source, f"Archived: {reason}")
            deal.updated_at = datetime.now().isoformat()

    def mark_junk(self, deal_id: str, reason: str = "", source: str = "manual"):
        """Mark a deal as junk"""
        deal = self.deals.get(deal_id)
        if deal:
            deal.status = "junk"
            deal.stage = "archive"
            deal.metadata.junk_reason = reason
            deal.metadata.archived_at = datetime.now().isoformat()
            deal.add_audit("marked_junk", source, f"Marked as junk: {reason}")
            deal.updated_at = datetime.now().isoformat()

    def merge_deals(self, source_deal_id: str, target_deal_id: str,
                    source: str = "manual") -> dict:
        """Merge source deal into target deal"""
        source_deal = self.deals.get(source_deal_id)
        target_deal = self.deals.get(target_deal_id)

        if not source_deal or not target_deal:
            return {"success": False, "error": "Deal not found"}

        # Transfer aliases
        aliases_merged = 0
        for alias in source_deal.aliases:
            if target_deal.add_alias(alias.alias, alias.alias_type, alias.confidence, "merge"):
                aliases_merged += 1

        # Transfer email thread IDs
        for thread_id in source_deal.email_thread_ids:
            if thread_id not in target_deal.email_thread_ids:
                target_deal.email_thread_ids.append(thread_id)

        # Add source folder to related folders
        if source_deal.folder_path and source_deal.folder_path not in target_deal.related_folders:
            target_deal.related_folders.append(source_deal.folder_path)

        # Update email mappings
        for msg_id, deal_id in list(self.email_to_deal.items()):
            if deal_id == source_deal_id:
                self.email_to_deal[msg_id] = target_deal_id

        # Update thread mappings
        for tid, deal_id in list(self.thread_to_deal.items()):
            if deal_id == source_deal_id:
                self.thread_to_deal[tid] = target_deal_id

        # Mark source as merged
        source_deal.status = "merged"
        source_deal.add_audit("merged", source, f"Merged into {target_deal_id}")

        # Update target audit
        target_deal.add_audit("received_merge", source, f"Received merge from {source_deal_id}")
        target_deal.updated_at = datetime.now().isoformat()

        return {
            "success": True,
            "aliases_merged": aliases_merged,
            "source_folder": source_deal.folder_path
        }

    def list_deals(self, stage: Optional[str] = None,
                   status: Optional[str] = None,
                   include_deleted: bool = False) -> List[Deal]:
        """List deals with optional filtering"""
        result = []
        for deal in self.deals.values():
            if not include_deleted and deal.deleted:
                continue
            if stage and deal.stage != stage:
                continue
            if status and deal.status != status:
                continue
            result.append(deal)
        return result

    def _remove_mappings_for_deal(self, deal_id: str) -> None:
        """Remove email/thread mappings pointing to a deleted deal."""
        self.email_to_deal = {k: v for k, v in self.email_to_deal.items() if v != deal_id}
        self.thread_to_deal = {k: v for k, v in self.thread_to_deal.items() if v != deal_id}

    def _add_mappings_for_deal(self, deal: Deal) -> None:
        """Restore thread mappings when a deal is restored."""
        for thread_id in (deal.email_thread_ids or []):
            self.thread_to_deal[thread_id] = deal.deal_id

    def mark_deal_deleted(self, deal_id: str, operator: str, *, reason: Optional[str] = None) -> bool:
        """
        Soft delete the deal so it no longer appears in lists.
        Returns True if the deal was marked deleted, False if it was already deleted or missing.
        """
        deal = self.get_deal(deal_id)
        if not deal or deal.deleted:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        deal.deleted = True
        deal.deleted_at = now_iso
        deal.deleted_by = operator
        deal.deleted_reason = reason
        deal.status = deal.status or 'deleted'
        deal.updated_at = now_iso
        deal.add_audit('deleted', operator, reason or 'deleted via UI')
        self._remove_mappings_for_deal(deal_id)
        return True

    def restore_deal(self, deal_id: str, operator: str, *, reason: Optional[str] = None) -> bool:
        """
        Restore a previously deleted deal and re-enable its thread mappings.
        """
        deal = self.get_deal(deal_id)
        if not deal or not deal.deleted:
            return False
        now_iso = datetime.now(timezone.utc).isoformat()
        deal.deleted = False
        deal.deleted_at = None
        deal.deleted_by = None
        deal.deleted_reason = None
        deal.status = deal.status or "active"
        deal.updated_at = now_iso
        deal.add_audit("restored", operator, reason or "restored via UI")
        self._add_mappings_for_deal(deal)
        return True

    def search(self, query: str) -> List[Dict]:
        """Search deals by query"""
        query_normalized = normalize_text(query)
        results = []

        for deal_id, deal in self.deals.items():
            score = 0
            matched_alias = None
            match_type = None

            # Check canonical name
            if query_normalized in normalize_text(deal.canonical_name):
                score = max(score, 0.9)
                matched_alias = deal.canonical_name
                match_type = "canonical_name"

            # Check listing IDs
            for lid in deal.identifiers.listing_ids:
                if query_normalized == lid or query in lid:
                    score = max(score, 1.0)
                    matched_alias = lid
                    match_type = "listing_id"

            # Check aliases
            for alias in deal.aliases:
                if query_normalized in alias.alias_normalized:
                    alias_score = 0.8 * alias.confidence
                    if alias_score > score:
                        score = alias_score
                        matched_alias = alias.alias
                        match_type = alias.alias_type

            if score > 0:
                results.append({
                    "deal_id": deal_id,
                    "canonical_name": deal.canonical_name,
                    "folder_path": deal.folder_path,
                    "stage": deal.stage,
                    "status": deal.status,
                    "matched_alias": matched_alias,
                    "match_type": match_type,
                    "confidence": score,
                })

        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results


# ============================================================================
# Migration Functions
# ============================================================================

def migrate_from_existing(
    registry: DealRegistry,
    dataroom_path: str,
    tracker_csv: str,
    broker_csv: str,
    dry_run: bool = False
) -> dict:
    """Migrate existing deals to registry"""
    dataroom = Path(dataroom_path)
    results = {
        "deals_created": 0,
        "aliases_created": 0,
        "brokers_created": 0,
        "junk_identified": 0,
        "duplicates": [],
    }

    # Load broker tracker
    brokers_by_email = {}
    if Path(broker_csv).exists():
        with open(broker_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                email = row.get('Email', '').lower().strip()
                if email:
                    broker = BrokerInfo(
                        broker_id=registry.generate_broker_id() if not dry_run else "BRK-XXX",
                        name=row.get('Broker Name', '').strip(),
                        email=email,
                        phone=row.get('Phone', '').strip(),
                        company=row.get('Company', '').strip(),
                        quality_rating=row.get('Quality Rating', 'MEDIUM').strip(),
                        sectors=[s.strip() for s in row.get('Sectors', '').split(',') if s.strip()],
                    )
                    brokers_by_email[email] = broker
                    if not dry_run:
                        registry.brokers[broker.broker_id] = broker
                    results["brokers_created"] += 1

    # Load master deal tracker
    if Path(tracker_csv).exists():
        with open(tracker_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                deal_name = row.get('Deal Name', '').strip()
                if not deal_name:
                    continue

                # Find broker
                broker_email = row.get('Broker Email', '').lower().strip()
                broker = brokers_by_email.get(broker_email)

                # Determine stage from folder location or tracker
                stage = row.get('Stage', 'New Inbound').lower()
                if 'screening' in stage or 'nda' in stage.lower():
                    stage = 'screening'
                elif 'qualified' in stage.lower() or 'loi' in stage.lower():
                    stage = 'qualified'
                else:
                    stage = 'inbound'

                # Extract listing IDs from status notes
                status_notes = row.get('Status Notes', '')
                listing_ids = re.findall(r'listing\s*[#:]?\s*(\d{4,8})', status_notes, re.IGNORECASE)
                listing_ids.extend(re.findall(r'\b(\d{7})\b', status_notes))

                # Generate deal ID
                deal_id = registry.generate_deal_id() if not dry_run else f"DEAL-XXXX-{results['deals_created']:03d}"

                if not dry_run:
                    deal = Deal(
                        deal_id=deal_id,
                        canonical_name=deal_name,
                        stage=stage,
                        status="active",
                        broker=broker,
                    )

                    # Add identifiers
                    deal.identifiers.listing_ids = list(set(listing_ids))

                    # Extract sector
                    sector = row.get('Sector', '').strip()
                    if sector and sector != 'TBD':
                        deal.company_info.sector = sector

                    # Add metadata
                    deal.metadata.priority = row.get('Priority', 'MEDIUM').strip()
                    deal.metadata.asking_price = row.get('Asking Price', '').strip()
                    deal.metadata.ebitda = row.get('EBITDA', '').strip()

                    # Generate aliases
                    deal.add_alias(normalize_text(deal_name), "name_variation", 1.0, "migration")
                    results["aliases_created"] += 1

                    for lid in listing_ids:
                        deal.add_alias(lid, "listing_number", 1.0, "migration")
                        deal.add_alias(f"listing {lid}", "listing_id", 1.0, "migration")
                        results["aliases_created"] += 2

                    deal.add_audit("created", "migration", f"Migrated from MASTER-DEAL-TRACKER.csv")
                    registry.deals[deal_id] = deal

                results["deals_created"] += 1

    # Scan inbound folders
    inbound_path = dataroom / "00-PIPELINE" / "Inbound"
    if inbound_path.exists():
        for folder in inbound_path.iterdir():
            if not folder.is_dir():
                continue

            folder_name = folder.name

            # Check if already in registry (by folder path)
            existing = registry.get_deal_by_folder(str(folder))
            if existing:
                continue

            # Check for junk patterns
            junk_patterns = [
                ("Holiday", "holiday promo"),
                ("Realtor", "realtor spam"),
                ("Construction-Listings", "realtor spam"),
                ("Most-Viewed", "realtor alerts"),
                ("Childs-Big-Feelings", "unrelated"),
            ]
            is_junk = False
            junk_reason = ""
            for pattern, reason in junk_patterns:
                if pattern.lower() in folder_name.lower():
                    is_junk = True
                    junk_reason = reason
                    results["junk_identified"] += 1
                    break

            if not dry_run and not existing:
                deal_id = registry.generate_deal_id()
                deal = Deal(
                    deal_id=deal_id,
                    canonical_name=folder_name,
                    folder_path=str(folder),
                    stage="inbound",
                    status="junk" if is_junk else "active",
                )

                if is_junk:
                    deal.metadata.junk_reason = junk_reason

                # Extract listing IDs from folder name
                listing_ids = re.findall(r'(\d{5,7})', folder_name)
                deal.identifiers.listing_ids = listing_ids

                deal.add_alias(normalize_text(folder_name), "name_variation", 1.0, "migration")
                for lid in listing_ids:
                    deal.add_alias(lid, "listing_number", 0.9, "migration")

                deal.add_audit("created", "migration", f"Migrated from Inbound folder")
                registry.deals[deal_id] = deal
                results["deals_created"] += 1

    # Scan screening folders
    screening_path = dataroom / "00-PIPELINE" / "Screening"
    if screening_path.exists():
        for folder in screening_path.iterdir():
            if not folder.is_dir():
                continue

            existing = registry.get_deal_by_folder(str(folder))
            if existing:
                existing.stage = "screening"
                continue

            if not dry_run:
                deal_id = registry.generate_deal_id()
                deal = Deal(
                    deal_id=deal_id,
                    canonical_name=folder.name,
                    folder_path=str(folder),
                    stage="screening",
                    status="active",
                )
                deal.add_alias(normalize_text(folder.name), "name_variation", 1.0, "migration")
                deal.add_audit("created", "migration", f"Migrated from Screening folder")
                registry.deals[deal_id] = deal
                results["deals_created"] += 1

    # Add default junk patterns
    default_junk = [
        ("realtor.com", "domain"),
        ("holiday.*shipping", "subject_regex"),
        ("construction listings update", "subject_contains"),
        ("most viewed", "subject_contains"),
    ]
    for pattern, ptype in default_junk:
        if not dry_run:
            registry.add_junk_pattern(pattern, ptype, "reject", "Default migration pattern")

    if not dry_run:
        registry.save()

    return results


# For testing
if __name__ == "__main__":
    # Quick test
    registry = DealRegistry("/home/zaks/DataRoom/.deal-registry/deal_registry.json")
    print(f"Loaded {len(registry.deals)} deals")
