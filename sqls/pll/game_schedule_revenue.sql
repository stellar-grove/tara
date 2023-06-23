create view pll.game_schedule_revenue as with base as
    ( select s.game_date,
             s.team1,
             s.team2,
             s.Venue,
             s.City,
             s.Attendance,
             round(s.attendance * s.TktTier1 * s.PxTier1,2) tier1Revenue,
             round(s.attendance * s.TktTier2 * s.PxTier2,2) tier2Revenue,
             round(s.attendance * s.TktTier3 * s.PxTier3,2) tier3Revenue
     from pll.schedule s )
select *,
       round(base.tier1Revenue + base.tier2Revenue + base.tier3Revenue,2) totalRevenue
from base