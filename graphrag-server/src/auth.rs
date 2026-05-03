//! Authentication and Authorization Middleware
//!
//! **STATUS: Not yet ported to actix-web (issue #40).** The HTTP-glue
//! parts of this module still import from `axum::{extract, http,
//! middleware, response}` while the rest of the server is on actix-web.
//! The `auth` Cargo feature pulls `axum` in as an optional dep so the
//! module compiles and its security-critical logic (JWT issuance/
//! validation, RBAC, rate limiting, secret loading) can be unit-tested
//! ahead of the port. The auth routes are still NOT wired into
//! `main.rs` (`/auth/*` is commented out), so toggling the feature
//! does not expose any new HTTP surface.

use axum::{
    extract::{Extension, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Canonical `iss` claim placed on every JWT we mint and required on
/// every JWT we accept (#31). Pinning a fixed issuer keeps tokens
/// minted by other services that happen to share our HS256 secret out
/// of our trust boundary.
pub const JWT_ISSUER: &str = "graphrag-server";

/// Canonical `aud` claim. Keeps tokens issued for a different API
/// (e.g. an admin console reusing the same secret) from being accepted
/// here (#31).
pub const JWT_AUDIENCE: &str = "graphrag-api";

/// Minimum acceptable JWT secret length, in bytes. HS256 signatures
/// are only as strong as the shared secret; sub-32-byte secrets are
/// brute-forceable with commodity hardware (#31).
pub const JWT_SECRET_MIN_BYTES: usize = 32;

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Issued at (timestamp)
    pub iat: u64,
    /// Expiration time (timestamp)
    pub exp: u64,
    /// Issuer — pinned to `JWT_ISSUER` on mint, required on validate (#31).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,
    /// Audience — pinned to `JWT_AUDIENCE` on mint, required on validate (#31).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// User role
    pub role: UserRole,
    /// Custom claims
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// User roles for RBAC
///
/// Ordering encodes the privilege hierarchy `Admin > User > Readonly >
/// Guest`, so `actual >= minimum` is the canonical permission check
/// (see `require_role`). `PartialOrd`/`Ord` are implemented manually
/// instead of derived because the `#[derive]` order would put `Admin`
/// at the bottom — and the variant declaration order is part of the
/// public surface (it's also serialized via `#[serde(rename_all)]`,
/// which we don't want to reshuffle just to satisfy a derive).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    /// Administrator with full access
    Admin,
    /// Regular user with read/write access
    User,
    /// Read-only user
    Readonly,
    /// Guest with limited access
    Guest,
}

impl UserRole {
    /// Numeric privilege rank. Higher == more privileged. Used by
    /// `PartialOrd`/`Ord` so callers can write `actual >= minimum`.
    fn rank(self) -> u8 {
        match self {
            UserRole::Admin => 3,
            UserRole::User => 2,
            UserRole::Readonly => 1,
            UserRole::Guest => 0,
        }
    }
}

impl PartialOrd for UserRole {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UserRole {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

/// API key structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub key: String,
    pub user_id: String,
    pub role: UserRole,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub rate_limit: RateLimit,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum requests per window
    pub max_requests: usize,
    /// Window duration in seconds
    pub window_seconds: u64,
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            max_requests: 1000,
            window_seconds: 3600, // 1 hour
        }
    }
}

/// Authentication state
#[derive(Clone)]
pub struct AuthState {
    /// JWT secret key
    jwt_secret: String,
    /// API keys storage
    api_keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    /// Rate limiting state: (user_id, (count, window_start))
    rate_limits: Arc<RwLock<HashMap<String, (usize, u64)>>>,
    /// Per-role rate limits applied on the JWT (Bearer) auth path.
    /// API-key callers carry their own `RateLimit`; JWT callers do not,
    /// so without this table the Bearer branch had no per-user ceiling
    /// at all (#32). Roles missing from the map fall back to
    /// `RateLimit::default()`.
    jwt_rate_limits: HashMap<UserRole, RateLimit>,
}

impl AuthState {
    /// Create a new authentication state
    ///
    /// # Arguments
    /// * `jwt_secret` - Secret key for JWT signing (should be 32+ characters)
    ///
    /// Prefer [`AuthState::try_new`] in production wiring; this
    /// constructor is kept for backwards compatibility and accepts any
    /// length, including the empty string. Tests still use it.
    pub fn new(jwt_secret: String) -> Self {
        Self {
            jwt_secret,
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
            jwt_rate_limits: HashMap::new(),
        }
    }

    /// Construct an `AuthState` while enforcing the minimum JWT secret
    /// length (`JWT_SECRET_MIN_BYTES`). Returns
    /// `AuthError::TokenGenerationFailed` with an operator-facing
    /// message that *does not* include the secret value (#31).
    pub fn try_new(jwt_secret: String) -> Result<Self, AuthError> {
        let len = jwt_secret.len();
        if len < JWT_SECRET_MIN_BYTES {
            return Err(AuthError::TokenGenerationFailed(format!(
                "JWT secret too short: {len} bytes; minimum is {JWT_SECRET_MIN_BYTES}"
            )));
        }
        Ok(Self::new(jwt_secret))
    }

    /// Override the JWT-path rate limit for a specific role.
    ///
    /// Used at startup (and in tests) to install per-role ceilings; the
    /// JWT branch of `extract_auth_user` consults this table.
    #[allow(dead_code)]
    pub fn set_jwt_rate_limit(&mut self, role: UserRole, limit: RateLimit) {
        self.jwt_rate_limits.insert(role, limit);
    }

    /// Look up the JWT-path rate limit for a role, falling back to the
    /// global `RateLimit::default()` (1000 req/hr).
    fn jwt_rate_limit_for(&self, role: UserRole) -> RateLimit {
        self.jwt_rate_limits.get(&role).cloned().unwrap_or_default()
    }

    /// Generate a JWT token
    ///
    /// # Arguments
    /// * `user_id` - User identifier
    /// * `role` - User role
    /// * `duration_hours` - Token validity duration in hours
    pub fn generate_token(
        &self,
        user_id: &str,
        role: UserRole,
        duration_hours: u64,
    ) -> Result<String, AuthError> {
        let now = unix_secs(std::time::SystemTime::now())?;

        let claims = Claims {
            sub: user_id.to_string(),
            iat: now,
            exp: now + (duration_hours * 3600),
            iss: Some(JWT_ISSUER.to_string()),
            aud: Some(JWT_AUDIENCE.to_string()),
            role,
            custom: HashMap::new(),
        };

        encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_bytes()),
        )
        .map_err(|e| AuthError::TokenGenerationFailed(e.to_string()))
    }

    /// Validate a JWT token. Requires `exp`, `iss`, `aud` and pins
    /// `iss == JWT_ISSUER`, `aud == JWT_AUDIENCE` (#31).
    pub fn validate_token(&self, token: &str) -> Result<Claims, AuthError> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_required_spec_claims(&["exp", "iss", "aud"]);
        validation.set_issuer(&[JWT_ISSUER]);
        validation.set_audience(&[JWT_AUDIENCE]);

        decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_bytes()),
            &validation,
        )
        .map(|data| data.claims)
        .map_err(|e| AuthError::InvalidToken(e.to_string()))
    }

    /// Create an API key
    pub async fn create_api_key(
        &self,
        user_id: &str,
        role: UserRole,
        rate_limit: Option<RateLimit>,
    ) -> Result<String, AuthError> {
        let key = format!("grag_{}", uuid::Uuid::new_v4());

        let api_key = ApiKey {
            key: key.clone(),
            user_id: user_id.to_string(),
            role,
            created_at: chrono::Utc::now().to_rfc3339(),
            expires_at: None,
            rate_limit: rate_limit.unwrap_or_default(),
        };

        self.api_keys.write().await.insert(key.clone(), api_key);

        Ok(key)
    }

    /// Validate an API key
    pub async fn validate_api_key(&self, key: &str) -> Result<ApiKey, AuthError> {
        let keys = self.api_keys.read().await;
        keys.get(key).cloned().ok_or(AuthError::InvalidApiKey)
    }

    /// Revoke an API key
    #[allow(dead_code)]
    pub async fn revoke_api_key(&self, key: &str) -> Result<(), AuthError> {
        let mut keys = self.api_keys.write().await;
        keys.remove(key).ok_or(AuthError::InvalidApiKey)?;
        Ok(())
    }

    /// Check rate limit for a user
    pub async fn check_rate_limit(
        &self,
        user_id: &str,
        limit: &RateLimit,
    ) -> Result<(), AuthError> {
        let now = unix_secs(std::time::SystemTime::now())?;

        let mut rate_limits = self.rate_limits.write().await;

        let (count, window_start) = rate_limits.entry(user_id.to_string()).or_insert((0, now));

        // Reset if window expired
        if now - *window_start >= limit.window_seconds {
            *count = 0;
            *window_start = now;
        }

        // Check limit
        if *count >= limit.max_requests {
            return Err(AuthError::RateLimitExceeded {
                max: limit.max_requests,
                window: limit.window_seconds,
            });
        }

        // Increment count
        *count += 1;

        Ok(())
    }
}

/// Convert a `SystemTime` to seconds since the UNIX epoch.
///
/// Returns `AuthError::TokenGenerationFailed` instead of panicking when
/// the clock is set before 1970 — both `generate_token` and
/// `check_rate_limit` previously called `.unwrap()` on the result and
/// would crash the worker thread (#45).
fn unix_secs(t: std::time::SystemTime) -> Result<u64, AuthError> {
    t.duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|e| {
            AuthError::TokenGenerationFailed(format!("system clock before UNIX epoch: {e}"))
        })
}

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Missing authorization header")]
    MissingAuthHeader,

    #[error("Invalid authorization format")]
    InvalidAuthFormat,

    #[error("Invalid token: {0}")]
    InvalidToken(String),

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Token generation failed: {0}")]
    TokenGenerationFailed(String),

    #[error("Insufficient permissions")]
    #[allow(dead_code)]
    InsufficientPermissions,

    #[error("Rate limit exceeded: {max} requests per {window} seconds")]
    RateLimitExceeded { max: usize, window: u64 },
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingAuthHeader => (StatusCode::UNAUTHORIZED, self.to_string()),
            AuthError::InvalidAuthFormat => (StatusCode::UNAUTHORIZED, self.to_string()),
            AuthError::InvalidToken(_) => (StatusCode::UNAUTHORIZED, self.to_string()),
            AuthError::InvalidApiKey => (StatusCode::UNAUTHORIZED, self.to_string()),
            AuthError::TokenGenerationFailed(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            },
            AuthError::InsufficientPermissions => (StatusCode::FORBIDDEN, self.to_string()),
            AuthError::RateLimitExceeded { .. } => {
                (StatusCode::TOO_MANY_REQUESTS, self.to_string())
            },
        };

        (status, message).into_response()
    }
}

/// Authenticated user information extracted from request
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AuthUser {
    pub user_id: String,
    pub role: UserRole,
}

/// Extract authenticated user from request headers
pub async fn extract_auth_user(
    auth_state: &AuthState,
    headers: &HeaderMap,
) -> Result<AuthUser, AuthError> {
    let auth_header = headers
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(AuthError::MissingAuthHeader)?;

    // Check for Bearer token (JWT)
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        let claims = auth_state.validate_token(token)?;
        // Apply per-role JWT rate limit (#32). Previously the Bearer
        // branch returned without ever counting the request, so a JWT
        // had no per-user ceiling — only API-key callers were limited.
        let limit = auth_state.jwt_rate_limit_for(claims.role);
        auth_state.check_rate_limit(&claims.sub, &limit).await?;
        return Ok(AuthUser {
            user_id: claims.sub,
            role: claims.role,
        });
    }

    // Check for API key
    if let Some(key) = auth_header.strip_prefix("ApiKey ") {
        let api_key = auth_state.validate_api_key(key).await?;

        // Check rate limit
        auth_state
            .check_rate_limit(&api_key.user_id, &api_key.rate_limit)
            .await?;

        return Ok(AuthUser {
            user_id: api_key.user_id,
            role: api_key.role,
        });
    }

    Err(AuthError::InvalidAuthFormat)
}

/// Authentication middleware for Axum
///
/// Extracts and validates authentication from request headers.
/// Supports both JWT tokens and API keys.
pub async fn auth_middleware(
    State(auth_state): State<Arc<AuthState>>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let user = extract_auth_user(&auth_state, &headers).await?;

    // Store user in request extensions
    request.extensions_mut().insert(user);

    Ok(next.run(request).await)
}

/// Require minimum role for a route
///
/// Use this middleware after auth_middleware to enforce role requirements.
#[allow(dead_code)]
pub async fn require_role(
    minimum_role: UserRole,
) -> impl Fn(
    axum::extract::Extension<AuthUser>,
    Request,
    Next,
) -> futures::future::BoxFuture<'static, Result<Response, AuthError>> {
    move |Extension(user): axum::extract::Extension<AuthUser>, request: Request, next: Next| {
        let minimum_role = minimum_role;
        Box::pin(async move {
            // Hierarchy: Admin > User > Readonly > Guest. `UserRole`'s
            // `Ord` impl encodes this directly, so the brittle 7-arm
            // match with a `_ => false` catch-all (#46) is gone.
            if user.role < minimum_role {
                return Err(AuthError::InsufficientPermissions);
            }

            Ok(next.run(request).await)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_jwt_token() {
        let auth_state = AuthState::new("test_secret_key_32_characters_long".to_string());

        let token = auth_state
            .generate_token("user123", UserRole::User, 24)
            .unwrap();
        let claims = auth_state.validate_token(&token).unwrap();

        assert_eq!(claims.sub, "user123");
        assert_eq!(claims.role, UserRole::User);
    }

    #[tokio::test]
    async fn test_api_key() {
        let auth_state = AuthState::new("test_secret".to_string());

        let key = auth_state
            .create_api_key("user123", UserRole::User, None)
            .await
            .unwrap();
        let api_key = auth_state.validate_api_key(&key).await.unwrap();

        assert_eq!(api_key.user_id, "user123");
        assert_eq!(api_key.role, UserRole::User);
    }

    // Pre-UNIX-epoch SystemTime values must surface as TokenGenerationFailed
    // instead of panicking via .unwrap() (regression for #45).
    #[test]
    fn unix_secs_returns_err_on_pre_epoch_clock() {
        let before_epoch = std::time::UNIX_EPOCH - std::time::Duration::from_secs(1);
        let result = unix_secs(before_epoch);
        assert!(
            matches!(result, Err(AuthError::TokenGenerationFailed(_))),
            "expected TokenGenerationFailed, got {result:?}"
        );
    }

    // Sanity check: a normal post-epoch SystemTime returns the seconds value.
    #[test]
    fn unix_secs_returns_seconds_for_post_epoch_clock() {
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let result = unix_secs(t).expect("post-epoch time must succeed");
        assert_eq!(result, 1_700_000_000);
    }

    // Exhaustive role-hierarchy check (#46): every (actual, minimum) pair
    // must satisfy `actual >= minimum` iff actual's privilege level is at
    // least minimum's. Hierarchy: Admin > User > Readonly > Guest.
    #[test]
    fn user_role_ord_covers_every_pair() {
        use UserRole::*;
        let all = [Admin, User, Readonly, Guest];
        // (actual, minimum, expected has_permission)
        let cases: &[(UserRole, UserRole, bool)] = &[
            (Admin, Admin, true),
            (Admin, User, true),
            (Admin, Readonly, true),
            (Admin, Guest, true),
            (User, Admin, false),
            (User, User, true),
            (User, Readonly, true),
            (User, Guest, true),
            (Readonly, Admin, false),
            (Readonly, User, false),
            (Readonly, Readonly, true),
            (Readonly, Guest, true),
            (Guest, Admin, false),
            (Guest, User, false),
            (Guest, Readonly, false),
            (Guest, Guest, true),
        ];
        // Sanity: we covered every (actual, minimum) pair.
        assert_eq!(cases.len(), all.len() * all.len());
        for (actual, minimum, expected) in cases {
            let got = actual >= minimum;
            assert_eq!(
                got, *expected,
                "role hierarchy mismatch: actual={actual:?}, minimum={minimum:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_rate_limit() {
        let auth_state = AuthState::new("test_secret".to_string());

        let limit = RateLimit {
            max_requests: 2,
            window_seconds: 60,
        };

        // First two requests should succeed
        auth_state
            .check_rate_limit("user123", &limit)
            .await
            .unwrap();
        auth_state
            .check_rate_limit("user123", &limit)
            .await
            .unwrap();

        // Third should fail
        let result = auth_state.check_rate_limit("user123", &limit).await;
        assert!(matches!(result, Err(AuthError::RateLimitExceeded { .. })));
    }

    // AuthState::try_new must reject a missing or under-32-byte JWT secret
    // (#31). Anything weaker than 32 bytes is brute-forceable for HS256 and
    // makes the previous hardcoded "graphrag_secret_key_change_in_production"
    // default a security landmine.
    #[test]
    fn try_new_rejects_short_secret() {
        // Use a distinctive non-English value so we can assert the secret
        // bytes don't leak into the operator-facing error.
        let secret = "xq8z!".to_string();
        let result = AuthState::try_new(secret.clone());
        let msg = match result {
            Err(AuthError::TokenGenerationFailed(s)) => s,
            Err(other) => panic!("expected TokenGenerationFailed, got {other:?}"),
            Ok(_) => panic!("short secret must reject"),
        };
        // Length and threshold should appear in the operator-facing message,
        // but the secret itself must NOT be logged.
        assert!(msg.contains("32"), "msg should cite the threshold: {msg}");
        assert!(!msg.contains(&secret), "secret value must not leak: {msg}");
    }

    // Empty secrets are also rejected, with no panic.
    #[test]
    fn try_new_rejects_empty_secret() {
        let result = AuthState::try_new(String::new());
        assert!(matches!(result, Err(AuthError::TokenGenerationFailed(_))));
    }

    // try_new accepts an exactly-32-byte secret.
    #[test]
    fn try_new_accepts_secret_at_threshold() {
        let secret = "a".repeat(32);
        assert!(
            AuthState::try_new(secret).is_ok(),
            "32-byte secret must be accepted"
        );
    }

    // Issued tokens carry the canonical iss/aud claims and validate end-to-end.
    #[test]
    fn generate_and_validate_token_round_trip_with_iss_aud() {
        let auth_state = AuthState::try_new("a".repeat(32)).expect("state");
        let token = auth_state
            .generate_token("u1", UserRole::User, 1)
            .expect("token");
        let claims = auth_state.validate_token(&token).expect("validate");
        assert_eq!(claims.iss.as_deref(), Some(JWT_ISSUER));
        assert_eq!(claims.aud.as_deref(), Some(JWT_AUDIENCE));
    }

    // A token with the wrong issuer must be rejected even if the signature
    // verifies — protects against same-secret tokens minted by another
    // service in the same trust boundary (#31).
    #[test]
    fn validate_token_rejects_wrong_issuer() {
        let auth_state = AuthState::try_new("a".repeat(32)).expect("state");
        // Hand-roll a token with a foreign `iss` but the right secret.
        let now = unix_secs(std::time::SystemTime::now()).unwrap();
        let claims = Claims {
            sub: "u1".into(),
            iat: now,
            exp: now + 3600,
            iss: Some("other-service".into()),
            aud: Some(JWT_AUDIENCE.into()),
            role: UserRole::User,
            custom: HashMap::new(),
        };
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret("a".repeat(32).as_bytes()),
        )
        .unwrap();
        let err = auth_state
            .validate_token(&token)
            .expect_err("foreign iss must reject");
        assert!(matches!(err, AuthError::InvalidToken(_)));
    }

    // A token with the wrong audience must be rejected even if the
    // signature verifies (#31).
    #[test]
    fn validate_token_rejects_wrong_audience() {
        let auth_state = AuthState::try_new("a".repeat(32)).expect("state");
        let now = unix_secs(std::time::SystemTime::now()).unwrap();
        let claims = Claims {
            sub: "u1".into(),
            iat: now,
            exp: now + 3600,
            iss: Some(JWT_ISSUER.into()),
            aud: Some("other-api".into()),
            role: UserRole::User,
            custom: HashMap::new(),
        };
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret("a".repeat(32).as_bytes()),
        )
        .unwrap();
        let err = auth_state
            .validate_token(&token)
            .expect_err("foreign aud must reject");
        assert!(matches!(err, AuthError::InvalidToken(_)));
    }

    // JWT-authenticated callers must be rate-limited too (#32). The Bearer
    // branch of `extract_auth_user` previously returned the user without
    // ever calling `check_rate_limit`, so a stolen JWT had no per-user
    // ceiling at all.
    #[tokio::test]
    async fn extract_auth_user_rate_limits_jwt_path() {
        let mut auth_state = AuthState::new("test_secret_key_32_characters_long".to_string());
        // Tighten the per-role JWT ceiling so the test doesn't have to
        // burn the default 1000 requests.
        auth_state.set_jwt_rate_limit(
            UserRole::User,
            RateLimit {
                max_requests: 2,
                window_seconds: 60,
            },
        );

        let token = auth_state
            .generate_token("jwt_user", UserRole::User, 1)
            .expect("token");

        let mut headers = axum::http::HeaderMap::new();
        headers.insert("Authorization", format!("Bearer {token}").parse().unwrap());

        // First two extractions succeed.
        extract_auth_user(&auth_state, &headers).await.expect("1st");
        extract_auth_user(&auth_state, &headers).await.expect("2nd");

        // Third must be rejected as RateLimitExceeded — not panic, not
        // silently pass.
        let result = extract_auth_user(&auth_state, &headers).await;
        assert!(
            matches!(result, Err(AuthError::RateLimitExceeded { .. })),
            "expected RateLimitExceeded, got {result:?}"
        );
    }
}
