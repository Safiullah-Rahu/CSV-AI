/** Response shapes of the organization / platform APIs (mirrors backend/app/api/v1). */

export type Page<T> = { items: T[]; total: number; limit: number; offset: number };

export type Permission = { key: string; category: string; description: string };

export type MemberRole = {
  assignment_id: string;
  role_id: string;
  key: string;
  name: string;
  scope_type: "org" | "workspace" | string;
  scope_id: string | null;
  expires_at: string | null;
};

export type Member = {
  user_id: string;
  email: string;
  full_name: string;
  status: string;
  user_status: string;
  title: string;
  department_id: string | null;
  department: string | null;
  roles: MemberRole[];
  teams: string[];
  joined_at: string;
  last_login_at: string | null;
  mfa_enabled: boolean;
};

export type Role = {
  id: string;
  key: string;
  name: string;
  description: string;
  is_system: boolean;
  rank: number;
  permissions: string[];
  assignment_count: number;
};

export type RoleAssignment = {
  id: string;
  role_id: string;
  user_id: string | null;
  team_id: string | null;
  scope_type: string;
  scope_id: string | null;
  expires_at: string | null;
  created_at: string;
};

export type Team = {
  id: string;
  name: string;
  description: string;
  department_id: string | null;
  workspace_id: string | null;
  created_at: string;
  member_ids: string[];
};

export type Department = { id: string; name: string; description: string; created_at: string; member_count: number };

export type Workspace = {
  id: string;
  name: string;
  description: string;
  created_at: string;
  member_count: number;
  kb_count: number;
  is_member: boolean;
};

export type Invitation = {
  id: string;
  email: string;
  role_id: string | null;
  team_id: string | null;
  workspace_id: string | null;
  status: string;
  expires_at: string;
  created_at: string;
};

export type Conversation = {
  id: string;
  title: string;
  user_id: string;
  kb_ids: string[];
  agent_id: string | null;
  archived: boolean;
  message_count: number;
  created_at: string;
  updated_at: string;
  own: boolean;
};

export type ConversationShare = { id: string; subject_type: "user" | "team" | "org"; subject_id: string | null };

export type ApiKey = {
  id: string;
  name: string;
  prefix: string;
  scopes: string[];
  created_by: string;
  rate_limit_per_minute: number | null;
  expires_at: string | null;
  revoked_at: string | null;
  last_used_at: string | null;
  created_at: string;
  active: boolean;
};

export type Webhook = {
  id: string;
  url: string;
  description: string;
  events: string[];
  enabled: boolean;
  failure_count: number;
  last_delivery_at: string | null;
  created_at: string;
};

export type WebhookDelivery = {
  id: string;
  event_type: string;
  status: string;
  attempts: number;
  response_status: number | null;
  error: string | null;
  created_at: string;
  delivered_at: string | null;
};

export type AuditEvent = {
  id: number;
  action: string;
  outcome: string;
  actor_user_id: string | null;
  actor_email: string | null;
  actor_api_key_id: string | null;
  resource_type: string | null;
  resource_id: string | null;
  ip: string | null;
  user_agent: string | null;
  request_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type ProviderKind = { kind: string; label: string; keyless: boolean; embedding_only: boolean; suggested_models: string[] };

export type AIProvider = {
  id: string;
  name: string;
  kind: string;
  kind_label: string;
  base_url: string | null;
  has_credentials: boolean;
  models: Record<string, unknown>[];
  default_model: string | null;
  default_embedding_model: string | null;
  settings: Record<string, unknown>;
  enabled: boolean;
  is_default: boolean;
  status: string;
  last_checked_at: string | null;
  last_error: string | null;
  suggested_models: string[];
  models_refreshed_at: string | null;
};

export type Plan = { key: string; name: string; limits: Record<string, number>; features: string[]; price_monthly_usd: number | null };

export type Billing = {
  billing_enabled: boolean;
  plan: string | null;
  plan_name: string | null;
  status: string | null;
  provider: string | null;
  limits: Record<string, number>;
  usage: Record<string, number>;
  plans: Plan[];
};

export type SupportGrant = { id: string; platform_user_id: string; reason: string; expires_at: string | null; revoked_at: string | null; created_at: string };

export type Session = { id: string; user_agent: string; ip: string; created_at: string; last_used_at: string; expires_at: string; current: boolean };
