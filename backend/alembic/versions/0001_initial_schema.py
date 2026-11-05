"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-11-05 16:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', sa.Enum('screener', 'clinician', 'admin', name='user_role'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # screenings table
    op.create_table(
        'screenings',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('pseudonym_id', sa.String(length=64), nullable=False),
        sa.Column('result_json', sa.Text(), nullable=False),
        sa.Column('model_version', sa.String(length=64), nullable=False),
        sa.Column('audit_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.Enum('pending_review', 'confirmed', 'overridden', name='screening_status'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_screenings_pseudonym_id'), 'screenings', ['pseudonym_id'], unique=False)

    # reviews table
    op.create_table(
        'reviews',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('screening_id', sa.String(length=36), nullable=False),
        sa.Column('clinician_id', sa.String(length=36), nullable=False),
        sa.Column('decision', sa.Enum('confirmed', 'overridden', name='review_decision'), nullable=False),
        sa.Column('final_grade', sa.Integer(), nullable=True),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['clinician_id'], ['users.id']),
        sa.ForeignKeyConstraint(['screening_id'], ['screenings.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_reviews_screening_id'), 'reviews', ['screening_id'], unique=False)
    op.create_index(op.f('ix_reviews_clinician_id'), 'reviews', ['clinician_id'], unique=False)


def downgrade() -> None:
    op.drop_table('reviews')
    op.drop_table('screenings')
    op.drop_table('users')
