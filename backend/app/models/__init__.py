"""ORM models. Importing this package registers every table on ``Base.metadata``."""

from app.models.ai import Agent, AIProviderConfig
from app.models.chat import Citation, Conversation, ConversationShare, Message, MessageFeedback
from app.models.connections import (
    ActivityEvent,
    Credential,
    DatabaseConnection,
    HealthCheck,
    Integration,
    Job,
    SqlQueryLog,
    SyncRun,
)
from app.models.knowledge import (
    Chunk,
    ChunkEmbedding,
    Dataset,
    Document,
    DocumentVersion,
    KBConfigRevision,
    KnowledgeBase,
)
from app.models.platform import (
    ApiKey,
    Approval,
    AuditEvent,
    Notification,
    Plan,
    Subscription,
    UsageRecord,
    Webhook,
    WebhookDelivery,
)
from app.models.rbac import ResourceGrant, Role, RoleAssignment, RolePermission, SupportAccessGrant
from app.models.tenancy import (
    Department,
    EmailToken,
    Invitation,
    Membership,
    Organization,
    Team,
    TeamMember,
    User,
    UserSession,
    Workspace,
    WorkspaceMember,
)

__all__ = [
    "AIProviderConfig", "ActivityEvent", "Agent", "ApiKey", "Approval", "AuditEvent", "Chunk",
    "ChunkEmbedding", "Citation", "Conversation", "ConversationShare", "Credential",
    "DatabaseConnection", "Dataset", "Department", "Document", "DocumentVersion", "EmailToken",
    "HealthCheck", "Integration", "Invitation", "Job", "KBConfigRevision", "KnowledgeBase",
    "Membership", "Message", "MessageFeedback", "Notification", "Organization", "Plan",
    "ResourceGrant", "Role", "RoleAssignment", "RolePermission", "SqlQueryLog", "Subscription",
    "SupportAccessGrant", "SyncRun", "Team", "TeamMember", "UsageRecord", "User", "UserSession",
    "Webhook", "WebhookDelivery", "Workspace", "WorkspaceMember",
]
