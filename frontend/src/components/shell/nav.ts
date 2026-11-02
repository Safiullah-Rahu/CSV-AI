import {
  Activity,
  BarChart3,
  Blocks,
  Bot,
  Cable,
  Database,
  FileText,
  KeyRound,
  LayoutDashboard,
  Library,
  MessageSquareText,
  MessagesSquare,
  Network,
  ScrollText,
  Settings,
  Shield,
  ShieldCheck,
  Table2,
  Users,
} from "lucide-react";

export type NavItem = {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  /** Shown only when the user holds this capability (anywhere for workspace-scoped ones). */
  perm?: string;
  anyOf?: string[];
  platform?: boolean;
};

export const NAV: { group: string; items: NavItem[] }[] = [
  {
    group: "Workspace",
    items: [
      { href: "/overview", label: "Overview", icon: LayoutDashboard, perm: "org.view" },
      { href: "/chat", label: "AI Chat", icon: MessageSquareText, perm: "kb.chat" },
      { href: "/conversations", label: "Conversations", icon: MessagesSquare, perm: "conv.view_own" },
      { href: "/agents", label: "Agents & Workflows", icon: Bot, perm: "agents.view" },
    ],
  },
  {
    group: "Knowledge",
    items: [
      { href: "/knowledge-bases", label: "Knowledge Bases", icon: Library, perm: "kb.view" },
      { href: "/documents", label: "Documents", icon: FileText, perm: "kb.view" },
      { href: "/data-sources", label: "Data Sources", icon: Table2, perm: "kb.view" },
    ],
  },
  {
    group: "Connectivity",
    items: [
      { href: "/connections", label: "Connections", icon: Network, perm: "org.view" },
      { href: "/database-connections", label: "Database Connections", icon: Database, perm: "db.view" },
      { href: "/integrations", label: "Integrations", icon: Cable, perm: "integrations.view" },
    ],
  },
  {
    group: "Insights",
    items: [{ href: "/analytics", label: "Analytics & Usage", icon: BarChart3, perm: "analytics.view" }],
  },
  {
    group: "Organization",
    items: [
      { href: "/team", label: "Team & Users", icon: Users, perm: "org.view" },
      { href: "/roles", label: "Roles & Permissions", icon: ShieldCheck, anyOf: ["admin.roles", "admin.permissions"] },
      { href: "/developers", label: "API & Developers", icon: KeyRound, anyOf: ["admin.api_keys", "admin.platform_settings"] },
      { href: "/audit", label: "Audit Logs", icon: ScrollText, perm: "admin.audit" },
      { href: "/settings", label: "Settings", icon: Settings },
    ],
  },
  {
    group: "Platform",
    items: [{ href: "/admin", label: "Administration", icon: Shield, platform: true }],
  },
];

export const ICONS = { Activity, Blocks };
