"""
Tests for XY routing algorithm with Y->X turn prevention.

Tests verify:
1. Basic XY routing direction computation (EAST, WEST, NORTH, SOUTH, LOCAL)
2. XY priority (X direction before Y direction)
3. Y->X turn prevention (deadlock avoidance)
"""

import pytest
from src.core.router import Direction, XYRouter
from src.core.flit import FlitFactory


class TestXYRoutingDirections:
    """Test basic XY routing direction computation."""

    def test_route_east_when_dest_x_greater(self, xy_router, single_flit_factory):
        """Flit destined for higher X should route EAST."""
        # Router at (2,2), dest at (4,2)
        flit = single_flit_factory(src=(0, 2), dest=(4, 2))

        output = xy_router.compute_output_port(flit)

        assert output == Direction.EAST

    def test_route_west_when_dest_x_smaller(self, xy_router, single_flit_factory):
        """Flit destined for lower X should route WEST."""
        # Router at (2,2), dest at (0,2)
        flit = single_flit_factory(src=(4, 2), dest=(0, 2))

        output = xy_router.compute_output_port(flit)

        assert output == Direction.WEST

    def test_route_north_when_same_x_dest_y_greater(self, xy_router, single_flit_factory):
        """Flit with same X, higher Y should route NORTH."""
        # Router at (2,2), dest at (2,3) - using valid y (0-3 for 5x4 mesh)
        flit = single_flit_factory(src=(2, 0), dest=(2, 3))

        output = xy_router.compute_output_port(flit)

        assert output == Direction.NORTH

    def test_route_south_when_same_x_dest_y_smaller(self, xy_router, single_flit_factory):
        """Flit with same X, lower Y should route SOUTH."""
        # Router at (2,2), dest at (2,0)
        flit = single_flit_factory(src=(2, 3), dest=(2, 0))

        output = xy_router.compute_output_port(flit)

        assert output == Direction.SOUTH

    def test_route_local_when_at_destination(self, xy_router, single_flit_factory):
        """Flit at destination should route to LOCAL."""
        # Router at (2,2), dest at (2,2)
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        output = xy_router.compute_output_port(flit)

        assert output == Direction.LOCAL


class TestXYPriority:
    """Test XY routing priority (X before Y)."""

    def test_x_priority_over_y_northeast(self, xy_router, single_flit_factory):
        """XY routing should prioritize EAST over NORTH when both needed."""
        # Router at (2,2), dest at (4,3) - needs both EAST and NORTH
        flit = single_flit_factory(src=(0, 0), dest=(4, 3))

        output = xy_router.compute_output_port(flit)

        # X first, so EAST (not NORTH)
        assert output == Direction.EAST

    def test_x_priority_over_y_northwest(self, xy_router, single_flit_factory):
        """XY routing should prioritize WEST over NORTH when both needed."""
        # Router at (2,2), dest at (0,3) - needs both WEST and NORTH
        flit = single_flit_factory(src=(4, 0), dest=(0, 3))

        output = xy_router.compute_output_port(flit)

        # X first, so WEST (not NORTH)
        assert output == Direction.WEST

    def test_x_priority_over_y_southeast(self, xy_router, single_flit_factory):
        """XY routing should prioritize EAST over SOUTH when both needed."""
        # Router at (2,2), dest at (4,0) - needs both EAST and SOUTH
        flit = single_flit_factory(src=(0, 3), dest=(4, 0))

        output = xy_router.compute_output_port(flit)

        # X first, so EAST (not SOUTH)
        assert output == Direction.EAST

    def test_x_priority_over_y_southwest(self, xy_router, single_flit_factory):
        """XY routing should prioritize WEST over SOUTH when both needed."""
        # Router at (2,2), dest at (0,0) - needs both WEST and SOUTH
        flit = single_flit_factory(src=(4, 3), dest=(0, 0))

        output = xy_router.compute_output_port(flit)

        # X first, so WEST (not SOUTH)
        assert output == Direction.WEST

    def test_y_direction_after_x_complete(self, xy_router, single_flit_factory):
        """After X is complete (same X), Y direction should be used."""
        # Router at (2,2), dest at (2,3) - X already aligned, needs NORTH
        flit = single_flit_factory(src=(2, 0), dest=(2, 3))

        output = xy_router.compute_output_port(flit)

        assert output == Direction.NORTH


class TestYXTurnPrevention:
    """
    Test Y->X turn prevention (deadlock avoidance).

    XY routing rule: Once a flit is traveling in Y direction (from NORTH or SOUTH),
    it cannot turn back to X direction (EAST or WEST).

    This prevents cyclic dependencies that could cause deadlock.
    """

    def test_no_east_turn_after_north(self, xy_router, single_flit_factory):
        """Y->X turn prevention: flit from NORTH cannot turn EAST."""
        # Router at (2,2), dest at (4,2) would normally go EAST
        # But flit came from NORTH (Y direction) - should be blocked
        flit = single_flit_factory(src=(0, 3), dest=(4, 2))

        output = xy_router.compute_output_port(flit, input_dir=Direction.NORTH)

        # Y->X turn should be blocked - return None
        assert output is None

    def test_no_west_turn_after_north(self, xy_router, single_flit_factory):
        """Y->X turn prevention: flit from NORTH cannot turn WEST."""
        # Router at (2,2), dest at (0,2) would normally go WEST
        flit = single_flit_factory(src=(4, 3), dest=(0, 2))

        output = xy_router.compute_output_port(flit, input_dir=Direction.NORTH)

        assert output is None

    def test_no_east_turn_after_south(self, xy_router, single_flit_factory):
        """Y->X turn prevention: flit from SOUTH cannot turn EAST."""
        # Router at (2,2), dest at (4,2) would normally go EAST
        flit = single_flit_factory(src=(0, 1), dest=(4, 2))

        output = xy_router.compute_output_port(flit, input_dir=Direction.SOUTH)

        assert output is None

    def test_no_west_turn_after_south(self, xy_router, single_flit_factory):
        """Y->X turn prevention: flit from SOUTH cannot turn WEST."""
        # Router at (2,2), dest at (0,2) would normally go WEST
        flit = single_flit_factory(src=(4, 0), dest=(0, 2))

        output = xy_router.compute_output_port(flit, input_dir=Direction.SOUTH)

        assert output is None


class TestAllowedTurns:
    """Test turns that ARE allowed by XY routing."""

    def test_north_after_north_allowed(self, xy_router, single_flit_factory):
        """Y->Y continuation is allowed: NORTH can continue NORTH."""
        # Router at (2,2), dest at (2,3) - continue NORTH
        flit = single_flit_factory(src=(2, 0), dest=(2, 3))

        output = xy_router.compute_output_port(flit, input_dir=Direction.NORTH)

        assert output == Direction.NORTH

    def test_south_after_south_allowed(self, xy_router, single_flit_factory):
        """Y->Y continuation is allowed: SOUTH can continue SOUTH."""
        # Router at (2,2), dest at (2,0) - continue SOUTH
        flit = single_flit_factory(src=(2, 3), dest=(2, 0))

        output = xy_router.compute_output_port(flit, input_dir=Direction.SOUTH)

        assert output == Direction.SOUTH

    def test_north_after_east_allowed(self, xy_router, single_flit_factory):
        """X->Y turn is allowed: EAST can turn to NORTH."""
        # Router at (2,2), X already aligned, dest at (2,3) needs NORTH
        flit = single_flit_factory(src=(0, 0), dest=(2, 3))

        output = xy_router.compute_output_port(flit, input_dir=Direction.EAST)

        assert output == Direction.NORTH

    def test_south_after_east_allowed(self, xy_router, single_flit_factory):
        """X->Y turn is allowed: EAST can turn to SOUTH."""
        # Router at (2,2), X aligned, dest at (2,0) needs SOUTH
        flit = single_flit_factory(src=(0, 3), dest=(2, 0))

        output = xy_router.compute_output_port(flit, input_dir=Direction.EAST)

        assert output == Direction.SOUTH

    def test_north_after_west_allowed(self, xy_router, single_flit_factory):
        """X->Y turn is allowed: WEST can turn to NORTH."""
        # Router at (2,2), X aligned, dest at (2,3) needs NORTH
        flit = single_flit_factory(src=(4, 0), dest=(2, 3))

        output = xy_router.compute_output_port(flit, input_dir=Direction.WEST)

        assert output == Direction.NORTH

    def test_south_after_west_allowed(self, xy_router, single_flit_factory):
        """X->Y turn is allowed: WEST can turn to SOUTH."""
        # Router at (2,2), X aligned, dest at (2,0) needs SOUTH
        flit = single_flit_factory(src=(4, 3), dest=(2, 0))

        output = xy_router.compute_output_port(flit, input_dir=Direction.WEST)

        assert output == Direction.SOUTH

    def test_local_after_north_allowed(self, xy_router, single_flit_factory):
        """Y->LOCAL is allowed: NORTH can deliver to LOCAL."""
        # Router at (2,2), dest at (2,2) - deliver to LOCAL
        flit = single_flit_factory(src=(2, 0), dest=(2, 2))

        output = xy_router.compute_output_port(flit, input_dir=Direction.NORTH)

        assert output == Direction.LOCAL

    def test_local_after_south_allowed(self, xy_router, single_flit_factory):
        """Y->LOCAL is allowed: SOUTH can deliver to LOCAL."""
        # Router at (2,2), dest at (2,2) - deliver to LOCAL
        flit = single_flit_factory(src=(2, 3), dest=(2, 2))

        output = xy_router.compute_output_port(flit, input_dir=Direction.SOUTH)

        assert output == Direction.LOCAL

    def test_any_direction_from_local(self, xy_router, single_flit_factory):
        """LOCAL can route to any direction."""
        # Test all four cardinal directions from LOCAL input

        # LOCAL -> EAST
        flit_e = single_flit_factory(src=(0, 2), dest=(4, 2))
        assert xy_router.compute_output_port(flit_e, Direction.LOCAL) == Direction.EAST

        # LOCAL -> WEST
        flit_w = single_flit_factory(src=(4, 2), dest=(0, 2))
        assert xy_router.compute_output_port(flit_w, Direction.LOCAL) == Direction.WEST

        # LOCAL -> NORTH
        flit_n = single_flit_factory(src=(2, 0), dest=(2, 3))
        assert xy_router.compute_output_port(flit_n, Direction.LOCAL) == Direction.NORTH

        # LOCAL -> SOUTH
        flit_s = single_flit_factory(src=(2, 3), dest=(2, 0))
        assert xy_router.compute_output_port(flit_s, Direction.LOCAL) == Direction.SOUTH


class TestRoutingWithoutInputDir:
    """Test routing when input_dir is not specified (backward compatibility)."""

    def test_routing_works_without_input_dir(self, xy_router, single_flit_factory):
        """Routing should work when input_dir is None (no Y->X check)."""
        flit = single_flit_factory(src=(0, 0), dest=(4, 2))

        # Without input_dir, should return EAST
        output = xy_router.compute_output_port(flit, input_dir=None)

        assert output == Direction.EAST

    def test_routing_default_no_input_dir(self, xy_router, single_flit_factory):
        """Default compute_output_port call should work."""
        flit = single_flit_factory(src=(0, 0), dest=(4, 3))

        # Call without input_dir argument at all
        output = xy_router.compute_output_port(flit)

        assert output == Direction.EAST
