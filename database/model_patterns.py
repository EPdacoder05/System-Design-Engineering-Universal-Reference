"""
SQLAlchemy Model Patterns and Best Practices

Apply to: Database models, ORM patterns, soft deletes, audit trails, relationships
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    select,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""
    pass


class UUIDMixin:
    """Mixin for UUID primary keys"""
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class AuditMixin(TimestampMixin):
    """
    Mixin for audit trail with timestamps and user tracking.
    
    Includes:
    - created_at: When the record was created
    - updated_at: When the record was last updated
    - created_by: User ID who created the record
    """
    
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )


class SoftDeleteMixin:
    """
    Mixin for soft delete functionality.
    
    Records are marked as deleted rather than physically removed.
    """
    
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        index=True,
    )
    
    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted"""
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """Mark record as deleted"""
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self) -> None:
        """Restore a soft deleted record"""
        self.deleted_at = None


# Many-to-many association table for User-Role relationship
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True),
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id"), primary_key=True),
    Column("assigned_at", DateTime, default=lambda: datetime.now(timezone.utc), nullable=False),
)


class User(Base, UUIDMixin, AuditMixin, SoftDeleteMixin):
    """
    User model with all best practice mixins.
    
    Features:
    - UUID primary key
    - Audit trail (created_at, updated_at, created_by)
    - Soft delete capability
    - Relationships with roles and audit logs
    - Composite indexes for common queries
    """
    __tablename__ = "users"
    
    # Basic fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    
    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    
    full_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )
    
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    
    # Relationships
    roles: Mapped[List["Role"]] = relationship(
        "Role",
        secondary=user_roles,
        back_populates="users",
        lazy="selectin",
    )
    
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )
    
    # Composite indexes for common query patterns
    __table_args__ = (
        # Composite index for active user lookups
        Index("idx_user_email_active", "email", "is_active"),
        Index("idx_user_username_active", "username", "is_active"),
        # Index for soft delete queries
        Index("idx_user_deleted_at", "deleted_at"),
        # Composite index for admin queries
        Index("idx_user_active_superuser", "is_active", "is_superuser"),
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"


class Role(Base, UUIDMixin, TimestampMixin):
    """
    Role model for RBAC (Role-Based Access Control).
    
    Many-to-many relationship with User model.
    """
    __tablename__ = "roles"
    
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    permissions: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="JSON string of permissions",
    )
    
    # Relationships
    users: Mapped[List["User"]] = relationship(
        "User",
        secondary=user_roles,
        back_populates="roles",
        lazy="selectin",
    )
    
    def __repr__(self) -> str:
        return f"<Role(id={self.id}, name={self.name})>"


class AuditLog(Base, UUIDMixin, TimestampMixin):
    """
    Audit log model for tracking user actions.
    
    One-to-many relationship with User model.
    """
    __tablename__ = "audit_logs"
    
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    
    resource_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    
    resource_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    
    details: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="JSON string with additional details",
    )
    
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),  # IPv6 max length
        nullable=True,
    )
    
    user_agent: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="audit_logs",
        lazy="selectin",
    )
    
    # Composite indexes for audit queries
    __table_args__ = (
        Index("idx_audit_user_action", "user_id", "action"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, user_id={self.user_id}, action={self.action})>"


# Query helper functions

async def get_active_users(session: AsyncSession) -> List[User]:
    """
    Get all active users (not soft deleted).
    
    Example of filtering out soft-deleted records.
    """
    stmt = select(User).where(
        User.is_active == True,
        User.deleted_at.is_(None),
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_user_with_roles(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> Optional[User]:
    """
    Get user with roles eagerly loaded.
    
    Example of relationship loading.
    """
    stmt = select(User).where(
        User.id == user_id,
        User.deleted_at.is_(None),
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def soft_delete_user(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> Optional[User]:
    """
    Soft delete a user.
    
    Example of soft delete pattern.
    """
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    
    if user:
        user.soft_delete()
        await session.commit()
    
    return user


async def restore_user(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> Optional[User]:
    """
    Restore a soft deleted user.
    
    Example of restoring soft-deleted records.
    """
    stmt = select(User).where(User.id == user_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    
    if user:
        user.restore()
        await session.commit()
    
    return user


async def get_user_audit_logs(
    session: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 100,
) -> List[AuditLog]:
    """
    Get audit logs for a user.
    
    Example of one-to-many relationship query.
    """
    stmt = (
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# Example usage
if __name__ == "__main__":
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    
    async def main():
        # Create engine and session
        engine = create_async_engine(
            "postgresql+asyncpg://user:password@localhost:5432/testdb",
            echo=True,
        )
        
        async_session = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )
        
        print("=== SQLAlchemy Model Patterns Example ===\n")
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async with async_session() as session:
            # Create roles
            admin_role = Role(
                name="admin",
                description="Administrator role",
                permissions='["read", "write", "delete"]',
            )
            user_role = Role(
                name="user",
                description="Regular user role",
                permissions='["read"]',
            )
            session.add_all([admin_role, user_role])
            await session.commit()
            
            print("Created roles: admin, user\n")
            
            # Create users
            user1 = User(
                email="alice@example.com",
                username="alice",
                hashed_password="hashed_pwd_123",
                full_name="Alice Smith",
                is_active=True,
            )
            user1.roles.append(admin_role)
            
            user2 = User(
                email="bob@example.com",
                username="bob",
                hashed_password="hashed_pwd_456",
                full_name="Bob Jones",
                is_active=True,
            )
            user2.roles.append(user_role)
            
            session.add_all([user1, user2])
            await session.commit()
            
            print(f"Created users: {user1.username}, {user2.username}\n")
            
            # Create audit log
            audit_log = AuditLog(
                user_id=user1.id,
                action="login",
                resource_type="authentication",
                details='{"method": "password"}',
                ip_address="192.168.1.1",
            )
            session.add(audit_log)
            await session.commit()
            
            print(f"Created audit log for {user1.username}\n")
            
            # Query active users
            active_users = await get_active_users(session)
            print(f"Active users: {len(active_users)}")
            for user in active_users:
                print(f"  - {user.username} ({user.email})")
                print(f"    Roles: {[role.name for role in user.roles]}")
            print()
            
            # Soft delete user
            print(f"Soft deleting user: {user2.username}")
            await soft_delete_user(session, user2.id)
            
            # Query active users again
            active_users = await get_active_users(session)
            print(f"Active users after soft delete: {len(active_users)}")
            for user in active_users:
                print(f"  - {user.username}")
            print()
            
            # Restore user
            print(f"Restoring user: {user2.username}")
            await restore_user(session, user2.id)
            
            # Query active users again
            active_users = await get_active_users(session)
            print(f"Active users after restore: {len(active_users)}")
            print()
            
            # Get audit logs
            logs = await get_user_audit_logs(session, user1.id)
            print(f"Audit logs for {user1.username}: {len(logs)}")
            for log in logs:
                print(f"  - {log.action} on {log.resource_type} at {log.created_at}")
        
        # Cleanup
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        await engine.dispose()
    
    # Run example
    # asyncio.run(main())
    print("\nTo run this example:")
    print("1. Set up a PostgreSQL database")
    print("2. Update the database URL in main()")
    print("3. Uncomment asyncio.run(main())")
