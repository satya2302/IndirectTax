using IndirectTax.Data;
using IndirectTax.Models;
using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;

namespace IndirectTax.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class UsageController : ControllerBase
    {
        private readonly DataUsageRepository _repository;

        public UsageController(DataUsageRepository repository)
        {
            _repository = repository;
        }

        [HttpGet]
        public ActionResult<IEnumerable<DataUsage>> Get()
        {
            var data = _repository.GetAll();
            return Ok(data);
        }
    }
}
