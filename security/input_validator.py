"""
Input Validation & Attack Pattern Detection

Apply to: User input validation, API endpoints, form processing

Features:
- 23+ attack pattern detection via regex
- Input sanitization functions
- Content-type validation
- File upload validation (extension, MIME, magic bytes)
"""

import re
from typing import Optional, List, Dict
from enum import Enum


class AttackType(Enum):
    """Types of attacks detected."""
    SQL_INJECTION = "sql_injection"
    XSS_REFLECTED = "xss_reflected"
    XSS_STORED = "xss_stored"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"
    XXE_INJECTION = "xxe_injection"
    SSRF = "ssrf"
    HEADER_INJECTION = "header_injection"
    TEMPLATE_INJECTION = "template_injection"
    LOG_INJECTION = "log_injection"
    EMAIL_HEADER_INJECTION = "email_header_injection"
    UNICODE_ATTACK = "unicode_attack"
    NULL_BYTE_INJECTION = "null_byte_injection"


# ============================================================================
# Attack Pattern Regex Library (23+ patterns)
# ============================================================================


ATTACK_PATTERNS = {
    # SQL Injection - Multiple variants
    AttackType.SQL_INJECTION: [
        r"(\bUNION\b.*\bSELECT\b)",  # UNION-based
        r"(\bOR\b\s+[\d]+\s*=\s*[\d]+)",  # OR 1=1
        r"(\bAND\b\s+[\d]+\s*=\s*[\d]+)",  # AND 1=1
        r"(;\s*DROP\s+TABLE)",  # Drop table
        r"(;\s*DELETE\s+FROM)",  # Delete from
        r"(exec\s*\()",  # Exec statement
        r"(EXEC\s+sp_)",  # SQL Server stored procedures
        r"(xp_cmdshell)",  # SQL Server command execution
        r"('.*\bOR\b.*'=')",  # String-based OR
        r"(--\s*$)",  # SQL comment
        r"(/\*.*\*/)",  # SQL comment block
        r"(\bINTO\s+OUTFILE\b)",  # File writing
    ],
    
    # Cross-Site Scripting (XSS) - Reflected, Stored, DOM-based
    AttackType.XSS_REFLECTED: [
        r"(<script[^>]*>.*?</script>)",  # Script tags
        r"(javascript:)",  # JavaScript protocol
        r"(on\w+\s*=)",  # Event handlers (onclick, onerror, etc.)
        r"(<iframe[^>]*>)",  # Iframe injection
        r"(<object[^>]*>)",  # Object tag
        r"(<embed[^>]*>)",  # Embed tag
        r"(eval\s*\()",  # Eval function
        r"(expression\s*\()",  # CSS expression
    ],
    
    AttackType.XSS_STORED: [
        r"(<img[^>]*src[^>]*onerror)",  # Image with onerror
        r"(<svg[^>]*onload)",  # SVG with onload
        r"(<body[^>]*onload)",  # Body onload
        r"(document\.cookie)",  # Cookie stealing
        r"(document\.write)",  # DOM manipulation
    ],
    
    # Path Traversal / Directory Traversal
    AttackType.PATH_TRAVERSAL: [
        r"(\.\./)",  # Relative path up
        r"(\.\.\\)",  # Windows path up
        r"(%2e%2e/)",  # URL encoded ..
        r"(%252e%252e/)",  # Double URL encoded
        r"(/etc/passwd)",  # Linux sensitive file
        r"(/etc/shadow)",  # Linux password file
        r"(C:\\Windows\\)",  # Windows system path
        r"(\\\\)",  # UNC paths
    ],
    
    # Command Injection (OS command execution)
    AttackType.COMMAND_INJECTION: [
        r"(;\s*\w+)",  # Command chaining with semicolon
        r"(\|\s*\w+)",  # Pipe to another command
        r"(&\s*\w+)",  # Background execution
        r"(`.*`)",  # Backticks command substitution
        r"(\$\(.*\))",  # Command substitution
        r"(>\s*/dev/null)",  # Output redirection
        r"(2>&1)",  # Error redirection
        r"(&&\s*\w+)",  # Conditional execution
        r"(\|\|\s*\w+)",  # OR execution
    ],
    
    # LDAP Injection
    AttackType.LDAP_INJECTION: [
        r"(\*\))",  # LDAP wildcard
        r"(\(\|)",  # LDAP OR
        r"(\(&)",  # LDAP AND
        r"(\(!\()",  # LDAP NOT
        r"(\)\(cn=)",  # Filter injection
    ],
    
    # XML Injection
    AttackType.XML_INJECTION: [
        r"(<\?xml)",  # XML declaration
        r"(<!DOCTYPE)",  # DOCTYPE declaration
        r"(<!ENTITY)",  # Entity declaration
        r"(CDATA\[)",  # CDATA section
    ],
    
    # XXE (XML External Entity) Injection
    AttackType.XXE_INJECTION: [
        r"(<!ENTITY.*SYSTEM)",  # System entity
        r"(<!ENTITY.*PUBLIC)",  # Public entity
        r"(file://)",  # File protocol
        r"(php://filter)",  # PHP wrapper
    ],
    
    # Server-Side Request Forgery (SSRF)
    AttackType.SSRF: [
        r"(http://localhost)",  # Localhost access
        r"(http://127\.0\.0\.1)",  # Loopback IP
        r"(http://169\.254\.169\.254)",  # AWS metadata service
        r"(http://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})",  # Internal IPs
        r"(file:///)",  # File protocol
        r"(gopher://)",  # Gopher protocol
        r"(dict://)",  # Dict protocol
    ],
    
    # Header Injection (HTTP Response Splitting)
    AttackType.HEADER_INJECTION: [
        r"(\r\n|\n|\r)",  # CRLF injection
        r"(%0d%0a)",  # URL encoded CRLF
        r"(%0a)",  # URL encoded LF
        r"(%0d)",  # URL encoded CR
    ],
    
    # Template Injection (Jinja2, EL, etc.)
    AttackType.TEMPLATE_INJECTION: [
        r"(\{\{.*\}\})",  # Jinja2/Django template
        r"(\{%.*%\})",  # Template control structures
        r"(\$\{.*\})",  # Spring EL, FreeMarker
        r"(<%.*%>)",  # JSP, ERB
    ],
    
    # Log Injection (CRLF in logs)
    AttackType.LOG_INJECTION: [
        r"(\r\n.*\r\n)",  # Multi-line injection
        r"(%0d%0a.*%0d%0a)",  # URL encoded multi-line
    ],
    
    # Email Header Injection
    AttackType.EMAIL_HEADER_INJECTION: [
        r"(\nTo:)",  # Additional recipient
        r"(\nCc:)",  # Carbon copy
        r"(\nBcc:)",  # Blind carbon copy
        r"(\nSubject:)",  # Subject manipulation
        r"(%0aTo:)",  # URL encoded
        r"(%0aCc:)",  # URL encoded
    ],
    
    # Unicode Attacks (Normalization bypasses)
    AttackType.UNICODE_ATTACK: [
        r"(\\u[0-9a-fA-F]{4})",  # Unicode escape
        r"(%u[0-9a-fA-F]{4})",  # IIS Unicode
        r"(\\x[0-9a-fA-F]{2})",  # Hex escape
    ],
    
    # Null Byte Injection
    AttackType.NULL_BYTE_INJECTION: [
        r"(%00)",  # URL encoded null byte
        r"(\\x00)",  # Hex encoded null byte
        r"(\\0)",  # Octal encoded null byte
    ],
}


# ============================================================================
# Input Validation Functions
# ============================================================================


def detect_attack_patterns(input_string: str) -> List[Dict[str, str]]:
    """
    Detect attack patterns in input string.
    
    Apply to: API input validation, form validation, log analysis
    
    Args:
        input_string: User input to validate
    
    Returns:
        List of detected attacks with type and matched pattern
    
    Example:
        attacks = detect_attack_patterns("' OR '1'='1")
        if attacks:
            print(f"Detected: {attacks[0]['type']}")
    """
    detected = []
    
    for attack_type, patterns in ATTACK_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, input_string, re.IGNORECASE)
            if matches:
                detected.append({
                    "type": attack_type.value,
                    "pattern": pattern,
                    "matches": matches
                })
    
    return detected


def is_safe_input(input_string: str) -> bool:
    """
    Check if input is safe (no attack patterns detected).
    
    Apply to: Quick validation before processing
    
    Args:
        input_string: User input to validate
    
    Returns:
        True if safe, False if attack detected
    
    Example:
        if not is_safe_input(user_input):
            return {"error": "Invalid input"}
    """
    return len(detect_attack_patterns(input_string)) == 0


def sanitize_input(input_string: str, allow_html: bool = False) -> str:
    """
    Sanitize user input by removing/escaping dangerous characters.
    
    Apply to: Displaying user content, storing user input
    
    Args:
        input_string: User input to sanitize
        allow_html: If False, escape all HTML (default)
    
    Returns:
        Sanitized string
    
    Example:
        safe_text = sanitize_input("<script>alert('xss')</script>")
        # Returns: "&lt;script&gt;alert('xss')&lt;/script&gt;"
    """
    if not allow_html:
        # Escape HTML characters
        input_string = (input_string
                       .replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;")
                       .replace('"', "&quot;")
                       .replace("'", "&#x27;")
                       .replace("/", "&#x2F;"))
    
    # Remove null bytes
    input_string = input_string.replace("\x00", "")
    
    # Remove CRLF for header injection prevention
    input_string = input_string.replace("\r", "").replace("\n", "")
    
    return input_string


def sanitize_sql_input(input_string: str) -> str:
    """
    Sanitize input for SQL queries (use parameterized queries instead when possible).
    
    Apply to: Dynamic SQL queries (avoid if possible, use prepared statements)
    
    Args:
        input_string: Input for SQL query
    
    Returns:
        Sanitized string (escapes single quotes)
    
    Warning: Always prefer parameterized queries over this!
    
    Example:
        # BAD (SQL injection risk):
        # query = f"SELECT * FROM users WHERE name = '{sanitize_sql_input(user_input)}'"
        
        # GOOD (use parameterized queries):
        # query = "SELECT * FROM users WHERE name = ?"
        # cursor.execute(query, (user_input,))
    """
    # Escape single quotes
    return input_string.replace("'", "''")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal.
    
    Apply to: File uploads, file downloads
    
    Args:
        filename: Original filename
    
    Returns:
        Safe filename (no path components)
    
    Example:
        safe_name = sanitize_filename("../../etc/passwd")
        # Returns: "passwd"
    """
    # Remove path components
    filename = filename.split("/")[-1].split("\\")[-1]
    
    # Remove null bytes
    filename = filename.replace("\x00", "")
    
    # Remove dangerous characters
    filename = re.sub(r"[^\w\s\-\.]", "", filename)
    
    return filename


# ============================================================================
# Content-Type Validation
# ============================================================================


ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "text/html",
    "application/json",
    "application/xml",
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}


def validate_content_type(content_type: str, allowed_types: Optional[set] = None) -> bool:
    """
    Validate HTTP Content-Type header.
    
    Apply to: API endpoints, file uploads
    
    Args:
        content_type: Content-Type header value
        allowed_types: Set of allowed types (default: ALLOWED_CONTENT_TYPES)
    
    Returns:
        True if allowed
    
    Example:
        if not validate_content_type(request.headers.get("Content-Type")):
            return {"error": "Invalid content type"}
    """
    if allowed_types is None:
        allowed_types = ALLOWED_CONTENT_TYPES
    
    # Extract base content type (before semicolon)
    base_type = content_type.split(";")[0].strip().lower()
    
    return base_type in allowed_types


# ============================================================================
# File Upload Validation
# ============================================================================


# Magic bytes for common file types (first few bytes of file)
FILE_MAGIC_BYTES = {
    "jpg": [b"\xFF\xD8\xFF"],
    "png": [b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A"],
    "gif": [b"\x47\x49\x46\x38\x37\x61", b"\x47\x49\x46\x38\x39\x61"],
    "pdf": [b"%PDF"],
    "zip": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],
    "xml": [b"<?xml"],
}


ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}


def validate_file_extension(filename: str, allowed_extensions: Optional[set] = None) -> bool:
    """
    Validate file extension.
    
    Apply to: File uploads
    
    Args:
        filename: Uploaded filename
        allowed_extensions: Set of allowed extensions (default: ALLOWED_EXTENSIONS)
    
    Returns:
        True if extension allowed
    
    Example:
        if not validate_file_extension(uploaded_file.filename, {"png", "jpg"}):
            return {"error": "Invalid file type"}
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_EXTENSIONS
    
    if "." not in filename:
        return False
    
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions


def validate_file_magic_bytes(file_content: bytes, expected_type: str) -> bool:
    """
    Validate file by checking magic bytes (file signature).
    
    Apply to: File uploads (verify actual file type, not just extension)
    
    Args:
        file_content: First few bytes of file
        expected_type: Expected file type (e.g., "jpg", "png")
    
    Returns:
        True if magic bytes match
    
    Example:
        if not validate_file_magic_bytes(file_data[:20], "png"):
            return {"error": "File is not a valid PNG"}
    """
    if expected_type not in FILE_MAGIC_BYTES:
        return False
    
    for magic in FILE_MAGIC_BYTES[expected_type]:
        if file_content.startswith(magic):
            return True
    
    return False


def validate_uploaded_file(filename: str, file_content: bytes, max_size: int = 10 * 1024 * 1024) -> Dict:
    """
    Comprehensive file upload validation.
    
    Apply to: File upload endpoints
    
    Args:
        filename: Uploaded filename
        file_content: File content (bytes)
        max_size: Maximum file size in bytes (default: 10MB)
    
    Returns:
        Dict with "valid" (bool) and "errors" (list)
    
    Example:
        result = validate_uploaded_file(file.filename, file.read())
        if not result["valid"]:
            return {"errors": result["errors"]}
    """
    errors = []
    
    # Check filename
    safe_filename = sanitize_filename(filename)
    if not safe_filename:
        errors.append("Invalid filename")
    
    # Check extension
    if not validate_file_extension(filename):
        errors.append(f"File extension not allowed. Allowed: {ALLOWED_EXTENSIONS}")
    
    # Check size
    if len(file_content) > max_size:
        errors.append(f"File too large. Max: {max_size / 1024 / 1024}MB")
    
    if len(file_content) == 0:
        errors.append("File is empty")
    
    # Check magic bytes (if extension recognized)
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext in FILE_MAGIC_BYTES:
            if not validate_file_magic_bytes(file_content, ext):
                errors.append(f"File content does not match extension .{ext}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "safe_filename": safe_filename
    }


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    print("=== Input Validation & Attack Detection Demo ===\n")
    
    # Test cases
    test_inputs = [
        "normal input",
        "' OR '1'='1",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "test; rm -rf /",
        "http://169.254.169.254/latest/meta-data/",
        "{{7*7}}",
        "%00bypass.txt",
    ]
    
    for input_str in test_inputs:
        print(f"\nTesting: {input_str}")
        attacks = detect_attack_patterns(input_str)
        
        if attacks:
            print(f"  ⚠️  Attacks detected:")
            for attack in attacks:
                print(f"     - {attack['type']}")
        else:
            print("  ✅ Safe input")
        
        sanitized = sanitize_input(input_str)
        print(f"  Sanitized: {sanitized}")
    
    # File validation example
    print("\n\n=== File Validation Example ===")
    result = validate_uploaded_file("test.txt", b"Hello world", max_size=1024)
    print(f"Valid: {result['valid']}")
    if not result['valid']:
        print(f"Errors: {result['errors']}")
